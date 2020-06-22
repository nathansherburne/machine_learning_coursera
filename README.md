Usage:

1. Edit the Dockerfile `CMD` to specify the input file/arguments.
2. `docker build -t python:linear_regression .`
3. `docker run -it  -v $(pwd)/app/plots/:/app/plots python:linear_regression`
4. open the PDF file created.

