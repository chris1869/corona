import boto3
import json
import corona_sim
import os

sqs_url = os.getenv("SIM_QUEUE")
sqs_region = sqs_url.split("sqs.")[1].split(".amazonaws.")[0]
dynamo_table = os.getenv("SIM_TABLE")

sqs_client = boto3.client('sqs', region_name=sqs_region)

dydb_client = boto3.client("dynamodb", region_name=sqs_region)

scenario_types = dict(num_agents="N", height="N", width="N", R_spread="N", desease_duration="N", fps="N", agent_radius="N",
                      fatality="N", initial_sick="N", sd_impact="N", sd_start="N", sd_stop="N", sd_recovered="S", know_rate_sick="N", know_rate_recovered="N", party_freq="N", party_R_boost="N",
                      run="N", sim_md5="S")

result_types = dict(duration="N", sick_peak="N", death="N", infected="N", working="N", start_speed="N")

def gen_dynamo_msg(scenario, result):
    vals = {key: {scenario_types[key]: str(scenario[key])} for key in scenario.keys() if (not key in ["social_conf", "desease_conf"])}
    sconf = scenario["social_conf"]
    dconf = scenario["desease_conf"]
    vals.update({key: {scenario_types[key]: str(sconf[key])} for key in sconf.keys()})
    vals.update({key: {scenario_types[key]: str(dconf[key])} for key in dconf.keys()})
    vals.update({key: {result_types[key]: str(result[key])} for key in result.keys()})
    return vals

while True:
    response = sqs_client.receive_message(QueueUrl=sqs_url, MaxNumberOfMessages=10, WaitTimeSeconds=20)
    if not "Messages" in response:
        print("No more messages in queue. Shutting down")
        break

    for msg in response["Messages"]:
        print(msg["Body"])
        print(msg)

        scenario = json.loads(msg["Body"])
        try:
            result = corona_sim.runPyGame(scenario, None)
        except:
            result = dict(duration=-1, sick_peak=-1, death=-1, infected=-1, working=-1)
            print("Scenario malfunctioned")
            raise

        dyn_item = gen_dynamo_msg(scenario, result)
        dydb_client.put_item(TableName=dynamo_table, Item=dyn_item)
        respone = sqs_client.delete_message(QueueUrl=sqs_url, ReceiptHandle=msg["ReceiptHandle"])
