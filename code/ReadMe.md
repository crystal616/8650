### **ligtweightCoresets.py**

usage: `ligtweightCoresets.py [-h] [--seperator {'\t','\n',','}]
                             numOfVariable numOfCore filename`

description: Lightweight coreset construction

positional arguments:
  * `numOfVariable`        number of attributes
  * `numOfCore`           number of cores to use
  * `filename`             file name

optional arguments:
  * `-h, --help`           show this help message and exit
  * `--seperator {'\t','\n',','}`  the character used to seperate values
  
### **KMeans_eval.py**

usage: `KMeans_eval.py [-h] [--seperator {'\t','\n',','}] numOfVariable filename`

description: Run Kmean++ on both full dataset and sampled dataset and calculate quantization error

positional arguments:
  *`numOfVariable`        number of attributes
  *`filename`             filename

optional arguments:
  * `-h, --help`           show this help message and exit
  * `[--seperator {'\t','\n',','}]`  the character used to seperate values
