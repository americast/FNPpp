from matplotlib import pyplot as plt
from transformers.utils import send_example_telemetry
from tqdm import tqdm
from functools import lru_cache
from eval_metrics import rmse, nrmse, mape, crps_samples, get_pr

import pandas as pd
import numpy as np
send_example_telemetry("multivariate_informer_notebook", framework="pytorch")

from datasets import load_dataset
from functools import partial
from transformers import InformerConfig, InformerForPrediction
from pandas.core.arrays.period import period_array
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.time_feature import TimeFeature

from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.time_feature import get_lags_for_frequency

from gluonts.dataset.field_names import FieldName
from gluonts.transform import (
    AddAgeFeature,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AsNumpyArray,
    Chain,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    RemoveFields,
    SelectFields,
    SetField,
    TestSplitSampler,
    Transformation,
    ValidationSplitSampler,
    VstackFeatures,
    RenameFields,
)

from transformers import PretrainedConfig
from gluonts.transform.sampler import InstanceSampler
from typing import Optional

from typing import Iterable

from torch.utils.data import DataLoader

from gluonts.itertools import Cached, Cyclic, IterableSlice, PseudoShuffled
from gluonts.torch.util import IterableDataset

from accelerate import Accelerator
from torch.optim import AdamW
from evaluate import load
from gluonts.time_feature import get_seasonality
import matplotlib.dates as mdates

def run_traffic():
    dataset = load_dataset("monash_tsf", "traffic_weekly")

    train_example = dataset["train"][0]
    validation_example = dataset["validation"][0]

    freq = "1W"
    prediction_length = 8

    assert len(train_example["target"]) + prediction_length == len(
        dataset["validation"][0]["target"]
    )

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]



    @lru_cache(10_000)
    def convert_to_pandas_period(date, freq):
        return pd.Period(date, freq)


    def transform_start_field(batch, freq):
        batch["start"] = [convert_to_pandas_period(date, freq) for date in batch["start"]]
        return batch


    train_dataset.set_transform(partial(transform_start_field, freq=freq))
    test_dataset.set_transform(partial(transform_start_field, freq=freq))


    num_of_variates = len(train_dataset)

    train_grouper = MultivariateGrouper(max_target_dim=num_of_variates)
    test_grouper = MultivariateGrouper(
        max_target_dim=num_of_variates,
        num_test_dates=len(test_dataset) // num_of_variates, # number of rolling test windows
    )

    multi_variate_train_dataset = train_grouper(train_dataset)
    multi_variate_test_dataset = test_grouper(test_dataset)

    multi_variate_train_example = multi_variate_train_dataset[0]
    print("multi_variate_train_example[\"target\"].shape =", multi_variate_train_example["target"].shape)



    lags_sequence = get_lags_for_frequency(freq)
    print(lags_sequence)


    time_features = time_features_from_frequency_str(freq)
    print(time_features)


    timestamp = pd.Period("2015-01-01 01:00:01", freq=freq)
    timestamp_as_index = pd.PeriodIndex(data=period_array([timestamp]))
    additional_features = [
        (time_feature.__name__, time_feature(timestamp_as_index))
        for time_feature in time_features
    ]


    config = InformerConfig(
        # in the multivariate setting, input_size is the number of variates in the time series per time step
        input_size=num_of_variates,
        # prediction length:
        prediction_length=prediction_length,
        # context length:
        context_length=20,
        # lags value copied from 1 week before:
        lags_sequence=[1, 24 * 7],
        # we'll add 5 time features ("hour_of_day", ..., and "age"):
        num_time_features=len(time_features) + 1,
        
        # informer params:
        dropout=0.1,
        encoder_layers=6,
        decoder_layers=4,
        # project input from num_of_variates*len(lags_sequence)+num_time_features to:
        d_model=64,
    )

    model = InformerForPrediction(config)



    def create_transformation(freq: str, config: PretrainedConfig) -> Transformation:
        # create list of fields to remove later
        remove_field_names = []
        if config.num_static_real_features == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if config.num_dynamic_real_features == 0:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        if config.num_static_categorical_features == 0:
            remove_field_names.append(FieldName.FEAT_STATIC_CAT)

        return Chain(
            # step 1: remove static/dynamic fields if not specified
            [RemoveFields(field_names=remove_field_names)]
            # step 2: convert the data to NumPy (potentially not needed)
            + (
                [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_CAT,
                        expected_ndim=1,
                        dtype=int,
                    )
                ]
                if config.num_static_categorical_features > 0
                else []
            )
            + (
                [
                    AsNumpyArray(
                        field=FieldName.FEAT_STATIC_REAL,
                        expected_ndim=1,
                    )
                ]
                if config.num_static_real_features > 0
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    # we expect an extra dim for the multivariate case:
                    expected_ndim=1 if config.input_size == 1 else 2,
                ),
                # step 3: handle the NaN's by filling in the target with zero
                # and return the mask (which is in the observed values)
                # true for observed values, false for nan's
                # the decoder uses this mask (no loss is incurred for unobserved values)
                # see loss_weights inside the xxxForPrediction model
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                # step 4: add temporal features based on freq of the dataset
                # these serve as positional encodings
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=time_features_from_frequency_str(freq),
                    pred_length=config.prediction_length,
                ),
                # step 5: add another temporal feature (just a single number)
                # tells the model where in the life the value of the time series is
                # sort of running counter
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=config.prediction_length,
                    log_scale=True,
                ),
                # step 6: vertically stack all the temporal features into the key FEAT_TIME
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if config.num_dynamic_real_features > 0
                        else []
                    ),
                ),
                # step 7: rename to match HuggingFace names
                RenameFields(
                    mapping={
                        FieldName.FEAT_STATIC_CAT: "static_categorical_features",
                        FieldName.FEAT_STATIC_REAL: "static_real_features",
                        FieldName.FEAT_TIME: "time_features",
                        FieldName.TARGET: "values",
                        FieldName.OBSERVED_VALUES: "observed_mask",
                    }
                ),
            ]
        )




    def create_instance_splitter(
        config: PretrainedConfig,
        mode: str,
        train_sampler: Optional[InstanceSampler] = None,
        validation_sampler: Optional[InstanceSampler] = None,
    ) -> Transformation:
        assert mode in ["train", "validation", "test"]

        instance_sampler = {
            "train": train_sampler
            or ExpectedNumInstanceSampler(
                num_instances=1.0, min_future=config.prediction_length
            ),
            "validation": validation_sampler
            or ValidationSplitSampler(min_future=config.prediction_length),
            "test": TestSplitSampler(),
        }[mode]

        return InstanceSplitter(
            target_field="values",
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=config.context_length + max(config.lags_sequence),
            future_length=config.prediction_length,
            time_series_fields=["time_features", "observed_mask"],
        )



    def create_train_dataloader(
        config: PretrainedConfig,
        freq,
        data,
        batch_size: int,
        num_batches_per_epoch: int,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = True,
        **kwargs,
    ) -> Iterable:
        PREDICTION_INPUT_NAMES = [
            "past_time_features",
            "past_values",
            "past_observed_mask",
            "future_time_features",
        ]
        if config.num_static_categorical_features > 0:
            PREDICTION_INPUT_NAMES.append("static_categorical_features")

        if config.num_static_real_features > 0:
            PREDICTION_INPUT_NAMES.append("static_real_features")

        TRAINING_INPUT_NAMES = PREDICTION_INPUT_NAMES + [
            "future_values",
            "future_observed_mask",
        ]

        transformation = create_transformation(freq, config)
        transformed_data = transformation.apply(data, is_train=True)
        if cache_data:
            transformed_data = Cached(transformed_data)

        # we initialize a Training instance
        instance_splitter = create_instance_splitter(config, "train") + SelectFields(
            TRAINING_INPUT_NAMES
        )

        # the instance splitter will sample a window of
        # context length + lags + prediction length (from all the possible transformed time series, 1 in our case)
        # randomly from within the target time series and return an iterator.
        training_instances = instance_splitter.apply(
            Cyclic(transformed_data)
            if shuffle_buffer_length is None
            else PseudoShuffled(
                Cyclic(transformed_data),
                shuffle_buffer_length=shuffle_buffer_length,
            )
        )

        # from the training instances iterator we now return a Dataloader which will
        # continue to sample random windows for as long as it is called
        # to return batch_size of the appropriate tensors ready for training!
        return IterableSlice(
            iter(
                DataLoader(
                    IterableDataset(training_instances),
                    batch_size=batch_size,
                    **kwargs,
                )
            ),
            num_batches_per_epoch,
        )

    def create_test_dataloader(
        config: PretrainedConfig,
        freq,
        data,
        batch_size: int,
        **kwargs,
    ):
        PREDICTION_INPUT_NAMES = [
            "past_time_features",
            "past_values",
            "past_observed_mask",
            "future_time_features",
        ]
        if config.num_static_categorical_features > 0:
            PREDICTION_INPUT_NAMES.append("static_categorical_features")

        if config.num_static_real_features > 0:
            PREDICTION_INPUT_NAMES.append("static_real_features")

        transformation = create_transformation(freq, config)
        transformed_data = transformation.apply(data, is_train=False)

        # we create a Test Instance splitter which will sample the very last
        # context window seen during training only for the encoder.
        instance_sampler = create_instance_splitter(config, "test") + SelectFields(
            PREDICTION_INPUT_NAMES
        )

        # we apply the transformations in test mode
        testing_instances = instance_sampler.apply(transformed_data, is_train=False)

        # This returns a Dataloader which will go over the dataset once.
        return DataLoader(
            IterableDataset(testing_instances), batch_size=batch_size, **kwargs
        )

    train_dataloader = create_train_dataloader(
        config=config,
        freq=freq,
        data=multi_variate_train_dataset,
        batch_size=256,
        num_batches_per_epoch=100,
        num_workers=2,
    )

    test_dataloader = create_test_dataloader(
        config=config,
        freq=freq,
        data=multi_variate_test_dataset,
        batch_size=32,
    )



    epochs = 25
    loss_history = []

    accelerator = Accelerator()
    device = accelerator.device

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=1e-1)

    model, optimizer, train_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
    )

    model.train()
    for epoch in tqdm(range(epochs)):
        for idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(
                static_categorical_features=batch["static_categorical_features"].to(device)
                if config.num_static_categorical_features > 0
                else None,
                static_real_features=batch["static_real_features"].to(device)
                if config.num_static_real_features > 0
                else None,
                past_time_features=batch["past_time_features"].to(device),
                past_values=batch["past_values"].to(device),
                future_time_features=batch["future_time_features"].to(device),
                future_values=batch["future_values"].to(device),
                past_observed_mask=batch["past_observed_mask"].to(device),
                future_observed_mask=batch["future_observed_mask"].to(device),
            )
            loss = outputs.loss

            # Backpropagation
            accelerator.backward(loss)
            optimizer.step()

            loss_history.append(loss.item())
            if idx % 100 == 0:
                print(loss.item())

    # view training
    loss_history = np.array(loss_history).reshape(-1)
    x = range(loss_history.shape[0])
    plt.figure(figsize=(10, 5))
    plt.plot(x, loss_history, label="train")
    plt.title("Loss", fontsize=15)
    plt.legend(loc="upper right")
    plt.xlabel("iteration")
    plt.ylabel("nll")
    plt.savefig("informer loss.png")

    # Inference
    model.eval()

    forecasts_ = []

    for batch in test_dataloader:
        outputs = model.generate(
            static_categorical_features=batch["static_categorical_features"].to(device)
            if config.num_static_categorical_features > 0
            else None,
            static_real_features=batch["static_real_features"].to(device)
            if config.num_static_real_features > 0
            else None,
            past_time_features=batch["past_time_features"].to(device),
            past_values=batch["past_values"].to(device),
            future_time_features=batch["future_time_features"].to(device),
            past_observed_mask=batch["past_observed_mask"].to(device),
        )
        forecasts_.append(outputs.sequences.cpu().numpy())

    forecasts = np.vstack(forecasts_)
    print(forecasts.shape)



    mase_metric = load("evaluate-metric/mase")
    smape_metric = load("evaluate-metric/smape")

    forecast_median = np.median(forecasts, 1).squeeze(0).T

    mase_metrics = []
    smape_metrics = []
    rmse_metrics = []
    crps_metrics = []

    for item_id, ts in enumerate(test_dataset):
        training_data = ts["target"][:-prediction_length]
        ground_truth = ts["target"][-prediction_length:]
        mase = mase_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
            training=np.array(training_data),
            periodicity=get_seasonality(freq),
        )
        mase_metrics.append(mase["mase"])
        rmse_metrics.append(rmse(forecast_median[item_id][:3], ground_truth[:3]))
        crps_metrics.append(crps_samples(forecast_median[item_id][:3], ground_truth[:3]))
        smape = smape_metric.compute(
            predictions=forecast_median[item_id],
            references=np.array(ground_truth),
        )
        smape_metrics.append(smape["smape"])

    print(f"MASE: {np.mean(mase_metrics)}")
    print(f"sMAPE: {np.mean(smape_metrics)}")
    print(f"RMSE: {np.mean(rmse_metrics)}")
    print(f"CRPS: {np.mean(crps_metrics)}")
    plt.clf()
    plt.scatter(mase_metrics, smape_metrics, alpha=0.2)
    plt.xlabel("MASE")
    plt.ylabel("sMAPE")
    plt.savefig("informer eval.png")



    def plot(ts_index, mv_index):
        fig, ax = plt.subplots()

        index = pd.period_range(
            start=multi_variate_test_dataset[ts_index][FieldName.START],
            periods=len(multi_variate_test_dataset[ts_index][FieldName.TARGET]),
            freq=multi_variate_test_dataset[ts_index][FieldName.START].freq,
        ).to_timestamp()

        ax.xaxis.set_minor_locator(mdates.HourLocator())

        ax.plot(
            index[-2 * prediction_length :],
            multi_variate_test_dataset[ts_index]["target"][mv_index, -2 * prediction_length :],
            label="actual",
        )

        ax.plot(
            index[-prediction_length:],
            forecasts[ts_index, ..., mv_index].mean(axis=0),
            label="mean",
        )
        ax.fill_between(
            index[-prediction_length:],
            forecasts[ts_index, ..., mv_index].mean(0)
            - forecasts[ts_index, ..., mv_index].std(axis=0),
            forecasts[ts_index, ..., mv_index].mean(0)
            + forecasts[ts_index, ..., mv_index].std(axis=0),
            alpha=0.2,
            interpolate=True,
            label="+/- 1-std",
        )
        ax.legend()
        fig.autofmt_xdate()
        fig.savefig("informer plot.png")
    
    plot(0, 344)



if __name__ == '__main__':
    run_traffic()