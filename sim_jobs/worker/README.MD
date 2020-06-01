### Overall structure

* circ_diff.py: Helper functions to calculate collisions of agents - basically the physical simulation
* corona_sim.py: The core simulation script that simulates Social, Desease and Physical state of the simulation
* Dockerfile: Dockerfile to wrap worker into a docker container allowing execution on AWS (or other container platforms)
* requirements.txt: Defines the python libraries required by the container's python libraries
* worker.py: The core worker script that reads messages from the AWS SQS job queue, runs the simulation and writes results to AWS DynamoDB
* update_docker.sh: The script builds the docker container and registers the container on AWS Container Registry
