# Dependencies
import pandas as pd

# Name of the training file
file = "AI_Human.csv"


# Reads and separates data based on the pre-determined classifier
def read_data(file_name):
    with open(file_name) as text:
        data = pd.read_csv(text)

        # Separate data based on human/machine
        human = data[data["generated"] == 0]
        ai = data[data["generated"] == 1]

        print(human)
        print(ai)


def main():
    print("hello")
    read_data(file)


# Main()
if __name__ == "__main__":
    main()
