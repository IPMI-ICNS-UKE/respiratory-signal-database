from peewee import (
    BlobField,
    BooleanField,
    CharField,
    DateTimeField,
    FloatField,
    ForeignKeyField,
    IntegerField,
    Model,
    SqliteDatabase,
)

database = SqliteDatabase(None)


class BaseModel(Model):
    class Meta:
        database = database


class Patient(BaseModel):
    id = IntegerField(primary_key=True)
    origin = CharField()


class ResearchNumber(BaseModel):
    id = IntegerField(primary_key=True)
    patient = ForeignKeyField(Patient, backref="research_number")


class Signal(BaseModel):
    research_number = ForeignKeyField(ResearchNumber, backref="signal")
    modality = CharField()
    fraction = IntegerField()
    acquisition_date = DateTimeField()
    is_corrupted = BooleanField(default=False)
    df_signal = BlobField()
    length_secs = FloatField()
    hash = CharField(unique=True)


class RespiratoryStats(BaseModel):
    mean_period_secs = FloatField()
    std_period_secs = FloatField()
    mean_span_cm = FloatField()
    std_span_cm = FloatField()
    length_secs = FloatField()
    number_cycles = IntegerField()
    preprocessed = CharField()
    signal = ForeignKeyField(Signal, backref="respiratory_stats")


class DeepLearningDataset(BaseModel):
    project = CharField()
    set = CharField()  # train, val, test
    signal = ForeignKeyField(Signal, backref="meta_data")

    class Meta:
        indexes = (
            # unique combination
            (("project", "set", "signal"), True),
        )
