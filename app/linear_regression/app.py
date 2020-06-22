import os
import glob
import argparse
import pandas as pd
from plotnine import *
from gradient_descent import GradientDescent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-data', type=str, help="the input .csv file with columns of numeric training data")
    parser.add_argument('-e', '--exclude-columns', nargs='+', default=[], help="column names to not be used as x (i.e. input to the hypothesis function)")
    parser.add_argument('-x', '--include-only-columns', nargs='*', default=[], help="column names to use as x (i.e. inputs to different hypothesis functions)")
    parser.add_argument('-y', '--y-column', type=str, default=None, help="the name of the y column (i.e. the value that the hypothesis function solves for")
    parser.add_argument('-o', '--output_dir', type=str, help="the path of the output directory to save the output plots to")
    args = parser.parse_args()

    for f in get_files(args):
        print("Running gradient descent for file " + f)
        run_for_file(f, args.exclude_columns, args.include_only_columns, args.output_dir, args.y_column)

def get_files(args):
    input_path = args.input_data
    if os.path.isfile(input_path):
        files = [input_path]
    elif os.path.isdir(input_path):
        files = glob.glob(input_path + "*.csv")

    return files
    
def run_for_file(f, exclude_columns, include_only_columns, output_dir, y_column):
    file_data = pd.read_csv(f)
    # Make all column name references lower case from here on out for simplicity
    file_data.columns = [col.lower() for col in list(file_data)]
    x_columns, y = get_columns_to_run(file_data, exclude_columns, include_only_columns, y_column)
    plots = []
    for x in x_columns:
        print("Running gradient descent on " + x + " vs. " + y)
        data = file_data[[x, y]]
        gd = GradientDescent(data, 0.5)
        theta0, theta1 = gd.run()
        plots.append(get_plot(data, theta0, theta1, x, y))

    save_as_pdf_pages(plots, path=output_dir, dpi=100)

def get_columns_to_run(data, exclude, include_only, y):
    all_columns = list(data)
    include_only = [col.lower() for col in include_only]
    exclude = [col.lower() for col in exclude]
    y = y if y is not None else all_columns[-1]
    exclude.append(y)
    x_columns = []
    for col in all_columns:
        if col in exclude:
            continue
        if len(include_only) > 0 and col not in include_only:
            continue
        x_columns.append(col)

    return [col.lower() for col in x_columns], y.lower()

def get_plot(data, theta0, theta1, x, y):
    col_names = list(data)
    return ggplot(data) + geom_point(aes(x=x,y=y)) + geom_abline(intercept=theta0, slope=theta1)
    #p = p + expand_limits(x = 0, y = 0)

if __name__ == "__main__":
    main()
