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
  
### **CS.py**

usage: `CS.py [-h] [-coreset_number CORESET_NUMBER] [-seperator {'\t','\n',','}]
             [-start_row START_ROW] [-start_col START_COL] [-end_col END_COL]
             [-distance_func DISTANCE_FUNC]
             filename attributes cluster_number coreset_size`

Description: CS coreset construction

positional arguments:
  * `filename`             file name
  * `attributes`            number of attributes
  * `cluster_number`        number of cluster
  * `coreset_size`          coreset size

optional arguments:
  * `-h, --help `           show this help message and exit
  * `-coreset_number CORESET_NUMBER`
                        number of coresets to generate
  * `-seperator {'\t','\n',','}`    the character used to seperate values
  * `-start_row START_ROW`  the row at which the data start
  * `-start_col START_COL`  the column at which the data start
  * `-end_col END_COL`      the column at wich the data end
  * `-distance_func DISTANCE_FUNC`
                        function to calculate the distance
  
### **KMeans_eval.py**

usage: `KMeans_eval.py [-h] [--seperator {'\t','\n',','}] numOfVariable filename`

description: Run Kmean++ on both full dataset and sampled dataset and calculate quantization error

positional arguments:
  *`numOfVariable`        number of attributes
  *`filename`             filename

optional arguments:
  * `-h, --help`           show this help message and exit
  * `[--seperator {'\t','\n',','}]`  the character used to seperate values
