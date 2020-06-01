# PyGame template.
 
# Import standard modules.
import boto3
import hashlib
import json
import itertools
import random
import os

sqs_url = os.getenv("SIM_QUEUE")
sqs_region = sqs_url.split("sqs.")[1].split(".amazonaws.")[0]

client = boto3.client("sqs", region_name=sqs_region)

def get_social_conf(_sd_impact, _sd_start, _sd_stop, _sd_recovered, _know_rate_sick, _know_rate_recovered, _party_freq, _party_R_boost):
    return dict(sd_impact=_sd_impact, sd_start=_sd_start, sd_stop=_sd_stop, sd_recovered=_sd_recovered, know_rate_sick=_know_rate_sick,
                   know_rate_recovered=_know_rate_recovered, party_freq=_party_freq, party_R_boost=_party_R_boost)

def get_deases_conf(_R_spread, _dd, _fatality, _initial_sick):
    return dict(R_spread=_R_spread, desease_duration=_dd, fatality=_fatality, initial_sick=_initial_sick)

def post_simjob(scenario):
    scenario["run"] = 2
    msg = json.dumps(scenario, sort_keys=True).encode("utf-8")
    md5 = hashlib.md5(msg).hexdigest()
    msg = msg[:-1] + (', "sim_md5": "%s"}' % md5).encode("utf-8")

    client.send_message(QueueUrl=sqs_url,
                        MessageBody=msg.decode("utf-8"))

def gen_social_opts():
    sd_impacts = [0.1, 0.3, 0.5, 0.7, 0.9]
    sd_starts = [0.05, 0.10, 0.15, 0.20, 0.25]
    sd_stops = [0.01, 0.04, 0.09, 0.14, 0.19, 0.24]

    sd_recovereds = [True, False]
    know_rate_sicks = [0.1, 0.5, 0.9, 1.0]
    know_rate_recs = [0.1, 0.5, 0.9, 1.0]
    party_freqs = [0, 7, 13]
    party_R_boosts = [1, 3]
    return list(itertools.product(*[sd_impacts, sd_starts, sd_stops, sd_recovereds, know_rate_sicks,
                                    know_rate_recs, party_freqs, party_R_boosts]))

def gen_desease_opts():
    R_spreads = [2.0, 2.1, 2.2, 2.3, 2.4]
    DDs = [8]
    fatalities = [0.06, 0.10, 0.14]
    initial_sicks = [5]
    return list(itertools.product(*[R_spreads, DDs, fatalities, initial_sicks]))


def gen_env_opts():
    agent_options = [500]
    size_options = [(600, 600)]
    return list(itertools.product(*[agent_options, size_options]))

social_opts = gen_social_opts()
desease_opts = gen_desease_opts()
env_opts = gen_env_opts()

random.shuffle(desease_opts)
random.shuffle(social_opts)
random.shuffle(env_opts)

scenarios = []
num_scen = 0

total_scens = len(social_opts) * len(desease_opts) * len(env_opts)
print("Generating: %i scenarios" % total_scens)

for sopt in social_opts:
    if sopt[1] > sopt[2]:
        social_conf = get_social_conf(*sopt)
        for dopt in desease_opts:
            desease_conf = get_deases_conf(*dopt)
            for eopt in env_opts:
                scene = dict(num_agents=eopt[0], height=eopt[1][0], width=eopt[1][1], fps=10, agent_radius=3,
                                social_conf=social_conf, desease_conf=desease_conf)
                post_simjob(scene)
        print("Finished one social opt")

