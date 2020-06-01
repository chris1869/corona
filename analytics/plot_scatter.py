import pandas
import plotly.express as px
import numpy as np

df = pandas.read_pickle("stats.xz")

#print(df2.columns)

#df = df2.loc[df2["DESEASE_DURATION"] == 10,:]
#fig = px.scatter(x=df["DURATION"], y=df["INFECTED"], color=df["SOCIAL_ONSET"])
#fig = px.scatter(x=df["DEATH"], y=df["WORKING"], color=df["WIDTH"])

#print(df["DURATION"].median())
#calib_working = ((80 - df["DURATION"]) * df["DEATH"]) * df["FPS"] + df["WORKING"]

#i = np.argmax(df["DURATION"].values)
#print(df.iloc[i])

#fig = px.scatter(y=df["DURATION"], x=df["DEATH"], color=(df["SD_RECOVERED"]+1)*50) #color=df["SD_IMPACT"]*df["KNOW_RATE_SICK"])

maxd = np.max(df["DURATION"].values)
df["WORKING"] = (maxd - df["DURATION"]) * df["DEATH"] * df["FPS"] + df["WORKING"]
df["WORKING"] = np.log10(df["WORKING"]/(maxd*df["FPS"]*df["NUM_AGENTS"]))
#df["DURATION"] = np.log(df["DURATION"])
#df["INFECTED"] = np.log(df["INFECTED"]+1)
#fig = px.density_heatmap(df, x="WORKING", y="SD_IMPACT", marginal_x="histogram", marginal_y="histogram") #SPANNEND
#fig = px.density_heatmap(df.iloc[df.index[df["SD_RECOVERED"] == False]], x="WORKING", y="SICK_PEAK", marginal_x="histogram", marginal_y="histogram") #SPANNEND
#fig = px.density_heatmap(df, x="R_SPREAD", y="START_SPEED", marginal_x="histogram", marginal_y="histogram") #ALL GOOD
#fig = px.density_heatmap(df, x="SD_IMPACT", y="SD_STOP", marginal_x="histogram", marginal_y="histogram") #ALL GOOD

target = np.logical_and(np.logical_and(df["WORKING"] < 7, df["SICK_PEAK"] < 0.5), df["DEATH"] < 0.15)

d_bins = 20
w_bins = 20

w_steps = np.linspace(df["WORKING"].values.min(), df["WORKING"].values.max(), w_bins)
d_steps = np.linspace(df["DEATH"].values.min(), df["DEATH"].values.max(), d_bins)

hist = np.ones((d_bins-1, w_bins-1))
for i in range(d_bins-1):
    for k in range(w_bins-1):
        sub_inds = np.logical_and(np.logical_and(w_steps[k] <= df["WORKING"], df["WORKING"] < w_steps[k+1]), 
                                  np.logical_and(d_steps[i] <= df["DEATH"], df["DEATH"] < d_steps[i+1]))

        if np.any(sub_inds):
            hist[i, k] = np.mean(df.iloc[df.index[sub_inds]]["DURATION"].values)
        else:
            hist[i, k] = np.nan

import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
                   z=hist,
                   x=w_steps,
                   y=d_steps,
                   hoverongaps = False))
#fig = px.imshow(hist
                #labels=dict(x="Log(Work days lost)", y="Death rate", color="Minimal sick peak [%]"),
                #x=w_steps,
                #y=d_steps
#               )
#fig = px.scatter(x=df["WORKING"], y=df["DEATH"], color= target) #df["SICK_PEAK"]) #df["KNOW_RATE_SICK"]*df["SD_IMPACT"]) #color=df["SD_IMPACT"]*df["KNOW_RATE_SICK"])
#fig = px.scatter_3d(df.iloc[df.index[target]], x="WORKING", y="DEATH", z="SICK_PEAK") #, color=target) #df["SICK_PEAK"]) #df["KNOW_RATE_SICK"]*df["SD_IMPACT"]) #color=df["SD_IMPACT"]*df["KNOW_RATE_SICK"])


fig.show()
