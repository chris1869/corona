# PyGame template.
 
# Import standard modules.
import boto3
import hashlib
import json
import itertools
import random
import numpy as np
import os

sqs_url = os.getenv("SIM_QUEUE")
print(sqs_url)
print(os.environ["SIM_QUEUE"])
sqs_region = sqs_url.split("sqs.")[1].split(".amazonaws.")[0]

client = boto3.client("sqs", region_name=sqs_region)


def get_social_conf(_sd_impact, _sd_start, _sd_stop, _sd_recovered, _know_rate_sick, _know_rate_recovered, _party_freq, _party_R_boost):
    return dict(sd_impact=_sd_impact, sd_start=_sd_start, sd_stop=_sd_stop, sd_recovered=_sd_recovered, know_rate_sick=_know_rate_sick,
                   know_rate_recovered=_know_rate_recovered, party_freq=_party_freq, party_R_boost=_party_R_boost)

def get_deases_conf(_R_spread, _dd, _fatality, _initial_sick):
    return dict(R_spread=_R_spread, desease_duration=_dd, fatality=_fatality, initial_sick=_initial_sick)

def post_simjob(scenario, run_nr, iter_num):
    try:
        scenario["run"] = run_nr
        hash_dict = dict(scenario)
        msg = json.dumps(hash_dict, sort_keys=True).encode("utf-8")
        md5 = hashlib.md5(msg + b" iter : %i" % iter_num).hexdigest()
        msg = msg[:-1] + (', "sim_md5": "%s"}' % md5).encode("utf-8")
    except:
        print(scenario, run_nr, iter_num)
        raise

    client.send_message(QueueUrl=sqs_url,
                        MessageBody=msg.decode("utf-8"))

def gen_social_opts():
    sd_impacts = np.arange(0.1, 1.0, 0.1)
    sd_starts = np.arange(0.05, 0.30, 0.05) #[0.05, 0.10, 0.15, 0.20, 0.25]
    sd_stops = [0.01, 0.04, 0.09, 0.14, 0.19, 0.24]

    sd_recovereds = [True, False]
    know_rate_sicks = np.arange(0.1, 1.0, 0.1) #[0.1, 0.5, 0.9, 1.0]
    know_rate_recs = np.arange(0.1, 1.0, 0.1) #[0.1, 0.5, 0.9, 1.0]
    party_freqs = np.arange(1, 14, dtype=np.int32) #7, 13]
    party_R_boosts = np.arange(1, 5, dtype=np.int32)
    return dict(sd_impact=sd_impacts, sd_conf=list(itertools.product(*[sd_starts, sd_stops])),
                sd_recovered=sd_recovereds, know_rate_sick=know_rate_sicks,
                know_rate_recovered=know_rate_recs, party_conf=list(itertools.product(*[party_freqs, party_R_boosts])))

def gen_desease_opts():
    R_spreads = np.arange(2.0, 3.0)
    DDs = range(5, 15)
    fatalities = np.arange(0.05, 0.20, 0.01)
    initial_sicks = [5]
    return dict(R_spread=R_spreads, desease_duration=DDs, fatality=fatalities)

def gen_env_opts():
    agent_options = [500]
    size_options = [(600, 600)]
    return list(itertools.product(*[agent_options, size_options]))

social_opts = gen_social_opts()
desease_opts = gen_desease_opts()
env_opts = gen_env_opts()

scenarios = []
num_scen = 0

total_scens = len(social_opts) * len(desease_opts) # * len(env_opts)
print("Generating: %i scenarios" % total_scens)

base_social = get_social_conf(_sd_impact=0.9, _sd_start=0.05, _sd_stop=0.01, _sd_recovered=True, 
                              _know_rate_sick=0.9, _know_rate_recovered=0.1, _party_freq=0, _party_R_boost=1)


base_desease = get_deases_conf(_R_spread=2.3, _dd=8, _fatality=0.1, _initial_sick=5)
run_nr = 1003

for onum, opts in enumerate([social_opts, desease_opts]):
    for key in opts:
        s = dict(base_social)
        d = dict(base_desease)
        for v in opts[key]:
            t = s if onum == 0 else d
            if key == "sd_conf":
                if v[0] <= v[1]:
                    continue
                t["sd_start"] = v[0]
                t["sd_stop"] = v[1]
            elif key == "party_conf":
                t["party_freq"] = int(v[0])
                t["party_R_boost"] = int(v[1])
            else:
                t[key] = v

            for i in range(100):
                scene = dict(num_agents=500, height=600, width=600, fps=10, agent_radius=3,
                             social_conf=s, desease_conf=d)
                num_scen += 1
                post_simjob(scene, run_nr, i)
        run_nr += 1

for agents in range(500, 1000, 50):
    for density in [1., 0.9, 0.75, 1.1, 1.25]:
        scaling = np.sqrt(agents/500) * density
        for i in range(100):
            scene = dict(num_agents=agents, height=600*scaling, width=600*scaling, fps=10, agent_radius=3,
                        social_conf=base_social, desease_conf=base_desease)
            num_scen += 1
            post_simjob(scene, run_nr, i)
        if agents == 500 and density == 1.:
            print(run_nr)
        run_nr += 1

print(num_scen)
