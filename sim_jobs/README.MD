### Overall structure

* aws_tasks: Contains relevant scripts to create aws task defintions for executing the worker docker container on AWS Elastic Container Service (ECS) with FARGATE
* schedulers: Contains scheduler scripts that queue new simulation jobs within AWS SQS that can then be worked on by the workers
* worker: Contains the actual code to run the simulations, including a script to update the docker container on AWS Container Registry
