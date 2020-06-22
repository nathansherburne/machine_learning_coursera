FROM python:3
RUN pip install numpy
RUN pip install pandas
RUN pip install plotnine
ADD ./app /app/
WORKDIR /app
CMD ["python", "linear_regression/app.py", "-i", "/app/datasets/death_rates.csv", "-o", "/app/plots/", "-e", "index", "one"]
