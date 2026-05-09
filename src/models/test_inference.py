import mlflow.pyfunc

model = mlflow.pyfunc.load_model(
    "models:/gold-price-model@production"
)

print("Model Production berhasil dimuat")