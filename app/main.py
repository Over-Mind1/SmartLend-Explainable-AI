from src.utils.Data_Loader import data_loader

if __name__ == "__main__":
    df = data_loader()
    print(df.head())
