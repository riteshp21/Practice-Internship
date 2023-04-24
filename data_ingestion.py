import os, sys
from src.exception import CustomException
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('../../artifacts', 'train.csv')
    test_data_path: str = os.path.join('../../artifacts', 'test.csv')
    raw_data_path: str = os.path.join('../../artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        logging.info("Entered dataingestion component")

    def initiate_data_ingestion(self):
        try:
            df = pd.read_csv('D:\IntershipProject\SCMS_Delivery_History_Dataset.csv')
            logging.info("Read the dataset from dataframe")
            logging.info(f"No of columns{len(df.columns)} ")
            index1 = df[df["Freight Cost (USD)"].str.contains("Freight|Invoiced|See", case=False)].index
            index2 = df[df["Weight (Kilograms)"].str.contains("See|Weight")].index
            index3 = list(set(index1).union(set(index2)))
            df = df.drop(index3)
            df["Freight Cost (USD)"] = pd.to_numeric(df["Freight Cost (USD)"])
            df["Weight (Kilograms)"] = pd.to_numeric(df["Weight (Kilograms)"])
            df["Line Item Insurance (USD)"] = df["Line Item Insurance (USD)"].fillna(
                df["Line Item Insurance (USD)"].median())
            df = df.drop(['ID', 'Project Code', 'ASN/DN #', 'Delivered to Client Date', 'Delivery Recorded Date',
                          'Scheduled Delivery Date', 'Vendor INCO Term', 'PQ First Sent to Client Date'
                             , 'PO / SO #', 'PQ #', 'PO Sent to Vendor Date', 'Manufacturing Site',
                          'Molecule/Test Type', 'Item Description', 'Vendor','Dosage', 'Dosage Form',], axis=1)
            df["Shipment Mode"] = df["Shipment Mode"].fillna(df["Shipment Mode"].mode()[0])
            #df["Dosage"] = df["Dosage"].fillna(df["Dosage"].mode()[0])
            df.columns = df.columns.str.replace(" ", "_").str.replace("(", "").str.replace(")", "")
            logging.info(f"No of columns{len(df.columns)} ")

            numerical_columns = df.select_dtypes(exclude="object").columns
            categorical_columns = df.select_dtypes(include="object").columns

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train Test Initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Ingestion of data completed.")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))


