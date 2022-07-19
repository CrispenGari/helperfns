### helperfns

🎀 This is a python package that contains some helper functions for machine leaning.

<p align="center">
   <img src="https://github.com/CrispenGari/helperfns/blob/main/images/logo.png" alt="logo" width="60%"/>
</p>

### Getting started

To start using `helperfns` in your project you run the following command:

```shell
pip install helperfns
```

Or if you wan to install it in notebooks such as jupyter notebooks you can run the code cell with the following code:

```shell
!pip install helperfns
```

### Usage

The `helperfns` package is made up of different sub packages such as:

1. tables
2. text
3. utils
4. visualization

### tables

In the tables sub package you can print your data in tabular form for example:

```python
from helperfns.tables import tabulate_data

column_names = ["SUBSET", "EXAMPLE(s)", "Hello"]
row_data = [["training", 5, 4],['validation', 4, 4],['test', 3, '']]
tabulate_data(column_names, row_data)

```

Output:

```shell
+---------------------------------+
|              Table              |
+------------+------------+-------+
| SUBSET     | EXAMPLE(s) | Hello |
+------------+------------+-------+
| training   |          5 |     4 |
| validation |          4 |     4 |
| test       |          3 |       |
+------------+------------+-------+
```

### text

The text package offers two main function which are `clean_sentence` and `de_contract`

```python
from helperfns.text import clean_sentence, english_words, de_contract

# cleans the sentence
print(clean_sentence("text 1 # https://url.com/bla1/blah1/"))
# list of all english words
print(english_words)
# converts strings like `I'm` to 'I am'
print(de_contract("I'm"))
```

### utils

utils package comes with a simple helper function for converting seconds to hours, minutes and seconds.

Example:

```python
start = time.time()
for i in range(100000):
   pass
end = time.time()

```

Output:

```shell
'0:00:00.01'
```

### visualization

This sub package provides different helper functions for visualizing data using plots.

Examples:

```python
from helperfns.visualization import plot_complicated_confusion_matrix, plot_images, plot_images_predictions, plot_simple_confusion_matrix

# plot predicted image labels with the images
plot_images_predictions(images, true_labels, preds, classes=["dog", "cat"] ,cols=8)

# plot the images with their labels
plot_images(images[:24], true_labels[:24], cols=8)

# plot a simple confusion matrix
y_true = [random.randint(0, 1) for _ in range (100)]
y_pred = [random.randint(0, 1) for _ in range (100)]
classes =["dog", "cat"]
plot_simple_confusion_matrix(y_true, y_pred, classes)

# plot a confusion matrix with percentage value of confusion
y_true = [random.randint(0, 1) for _ in range (100)]
y_pred = [random.randint(0, 1) for _ in range (100)]
classes =["dog", "cat"]
plot_complicated_confusion_matrix(y_true, y_pred, classes)
```

### Contributing to `helperfns`.

To contribute to `helperfns` read the [CONTRIBUTION.md](https://github.com/CrispenGari/helperfns/blob/main/CONTRIBUTION.md) file.

### License

In this package the `MIT` license was used which reads as follows:

```
MIT License

Copyright (c) 2022 crispengari

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
