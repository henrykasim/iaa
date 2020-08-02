# IAA project


Features
--------

* Automated Machine Learning in Selecting best Predictive Model
* Make prediction based on the built predictive model



Starting the service
--------
Migrate core service database
> python manage.py migrate

Migrate core service database
> python manage.py migrate core

Start API service
> python manage.py runserver localhost:8000



Testing the API
--------
1. Submit PM Job given the dataset - return 2 top predictive model and set the automl runtime to 5 minutes (300 seconds)

> curl -X POST -F "file=@/path/to/file_train.csv" -F "uid=12345" -F "nModel=2" -F "maxRuntime=300" http://localhost:8000/runPM/


2. Get the result PM result from submitted uid 

> curl -X GET http://localhost:8000/pm/12345/


3. Upload Test dataset

> curl -X POST -F "file=@/path/to/file_test.csv" http://localhost:8000/pmdata/12345/



API Callable
--------

**/runPM/**
- @get: return all previously submitted predictive model
- @post: upload training dataset and build predictive model based on AutoML approach
parameter:
  * uid = unique identification
  * file = path to file
  * nModel = 1 (default 1)
  * maxRuntime = 300 (default 60)


**/pm/<uid>**
- @get: display the result of the submitted PM


**/pmdata/<uid>**
- @get: display dataset detail
- @post: submit test dataset
parameter:
  * file = path to file (without "target" column)
- @delete: remove dataset and the PM

