# Import standard modules.
import sys
import time
import numpy as np 

# Import non-standard modules.
import pandas as pd
from circ_diff import get_collisions, update_vectors, get_average_distance, unify_speed

desease_state = {"healthy": 1, "sick": 2, "recovered": 3, "dead": 4}
desease_color = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255), 4: (255, 125, 125)}
social_state = {"active": 0, "inactive": 1, "transparent": 2}

class Social_State():
    def __init__( self, num_agents, sd_impact=0.9, sd_start=0.02, sd_stop=0.01, sd_recovered=True,
                  know_rate_sick=0.7, know_rate_recovered=0.8,
                  party_freq=1, party_R_boost=3):

        self.num_agents = num_agents
        self.socials = None
        self.sd_active = None

        self.sd_impact = sd_impact
        self.sd_start = sd_start
        self.sd_stop = sd_stop

        self.know_rate_sick = know_rate_sick
        self.history = None

        self.know_rate_recovered = know_rate_recovered
        self.sd_recovered = sd_recovered #if true also recovered persons are socially distant

        self.party_freq = party_freq
        self.party_R_boost = party_R_boost

        self.active_agent_steps = 0
        self.sim_people_working = 0.
        self.sim_active = 0.
        self.init_state()


    def init_state(self):
        self.socials = np.ones(self.num_agents) * social_state["active"] #All are active
        self.sd_active = False
        self.social_count = 0
        self.history = {"active": []}
        self.update_KPIs(0)


    def update(self, dstate, dt):
        velo_update = None
        booster = None
        #Measure social distancing
        if self.sd_active:
            if (dstate.sim_sick * self.know_rate_sick) <= (self.sd_stop * self.num_agents):
                velo_update = np.arange(0, int(self.sd_impact*self.num_agents))
                self.socials[velo_update] = social_state["active"]
                booster = 1
                self.sd_active = False
        else:
            if (dstate.sim_sick * self.know_rate_sick) >= (self.sd_start * self.num_agents):
                self.sd_active = True
                self.socials[:int(self.sd_impact*self.num_agents)] = social_state["inactive"]

        #Measure recovered mobilization
        if self.sd_recovered:
            activate = np.logical_and(dstate.deseases == desease_state["recovered"], self.socials == social_state["inactive"])
            activate_inds = np.where(activate)[0]
            r = int(len(activate_inds) * self.know_rate_recovered)
            print("Reactivating: %i %.2f" % (r, self.know_rate_recovered))
            if r >= 1:
                velo_update = activate_inds[:r]
                self.socials[velo_update] = social_state["active"]
                booster = 1.

        #Measure infectuous Monday
        if self.party_freq > 0 and self.sd_active:
            if -dt < self.social_count < dt:
                num_distancing = int(self.sd_impact*self.num_agents)
                self.socials[:num_distancing] = social_state["active"]
                self.social_count -= dt
                velo_update = np.arange(0, num_distancing)
                booster = self.party_R_boost

            if self.social_count <= -1:
                self.social_count = self.party_freq
                self.socials[:int(self.sd_impact*self.num_agents)] = social_state["inactive"]

            if self.social_count != 0:
                self.social_count -= dt

        #All dead people are inactive
        self.socials[dstate.deseases == desease_state["dead"]] = social_state["inactive"]
        self.update_KPIs(dt)
        return velo_update, booster

    def print_stats(self, day):
        inactive = np.sum(self.socials == social_state["inactive"])
        print("Day ", day, "People_working [\%]: ", self.sim_people_working, inactive)

    def update_KPIs(self, dt):
        self.sim_active = np.sum(self.socials == social_state["inactive"])
        self.active_agent_steps += self.sim_active * dt
        self.sim_people_working = self.active_agent_steps
        #self.history["active"].append(self.sim_active)


class Desease_State():
    def __init__(self, num_agents, R_spread=2.2, desease_duration=10, fatality=0.1, initial_sick=1):
        self.num_agents = num_agents
        self.R_spread = R_spread
        self.desease_duration = desease_duration
        self.fatality = fatality
        self.initial_sick = initial_sick

        self.deseases = None
        self.sick = None
        self.peak_sick = 0
        self.sim_healthy = None
        self.sim_sick = None
        self.sim_recovered = None
        self.sim_dead = None
        self.history = None
        self.infects = None

        self.sim_sick_peak = 0.
        self.sim_people_died = 0.
        self.sim_people_infected = 0.
        self.init_state()


    def init_state(self):
        self.deseases = np.ones(self.num_agents) #All are healthy
        self.deseases[:self.initial_sick] = desease_state["sick"]

        self.sick = np.zeros(self.num_agents)
        self.sick[:self.initial_sick] = self.desease_duration
        #self.history = {"healthy":[], "sick":[], "recovered":[], "dead":[]}

        self.infects = np.zeros(self.num_agents)
        self.update_KPIs()

    def update(self, dt, inds1, inds2):
        healthy1 = self.deseases[inds1] == desease_state["healthy"]
        sick2 = self.deseases[inds2] == desease_state["sick"]

        sick1 = self.deseases[inds1] == desease_state["sick"]
        healthy2 = self.deseases[inds2] == desease_state["healthy"]

        infected2 = inds2[np.logical_and(healthy2,sick1)]
        self.deseases[infected2] = desease_state["sick"]
        self.sick[infected2] = self.desease_duration

        infected1 = inds1[np.logical_and(healthy1, sick2)]
        self.deseases[infected1] = desease_state["sick"]
        self.sick[infected1] = self.desease_duration

        self.infects[inds1[np.logical_and(healthy2, sick1)]] += 1
        self.infects[inds2[np.logical_and(healthy1, sick2)]] += 1

        sick_agents = self.sick > 0
        recovering_agents = np.logical_and((1-dt) < self.sick, self.sick < (1+ dt))
        self.sick[sick_agents] -= dt
        self.deseases[recovering_agents] = desease_state["recovered"] + (np.random.rand(np.sum(recovering_agents)) < self.fatality).astype(int)
        s = np.sum(self.deseases == desease_state["sick"])
        if s > self.peak_sick:
            self.peak_sick = s
        self.update_KPIs()

    def update_KPIs(self):
        self.sim_healthy = np.sum(self.deseases == desease_state["healthy"])
        self.sim_sick = np.sum(self.deseases == desease_state["sick"])
        self.sim_recovered = np.sum(self.deseases == desease_state["recovered"])
        self.sim_dead = np.sum(self.deseases == desease_state["dead"])

        self.sim_sick_peak = self.peak_sick/self.num_agents
        self.sim_people_died = np.sum(self.deseases == desease_state["dead"])/self.num_agents
        self.sim_people_infected = np.sum(self.deseases == desease_state["healthy"])/self.num_agents

        sick_inds = self.deseases == desease_state["sick"]
        self.sim_R_spread = np.mean(self.infects[sick_inds])#/(self.desease_duration - self.sick[sick_inds]))
        
        #self.history["healthy"].append(self.sim_healthy)
        #self.history["sick"].append(self.sim_sick)
        #self.history["recovered"].append(self.sim_recovered)
        #self.history["dead"].append(self.sim_dead)

    def print_stats(self, day):
        #sick_inds = self.deseases == desease_state["sick"]
        #print("R_spread: ", np.mean(self.infects[sick_inds]/(self.desease_duration - self.sick[sick_inds])))
        print("Day: ", day, "Healthy: ", self.sim_healthy, "Sick: ", self.sim_sick, "Recovered: ", self.sim_recovered,
                            "Dead: ", self.sim_dead) 


class Corona_Simulation():
    def __init__(self, num_agents, height, width, fps, social_conf, desease_conf, agent_radius = 3,run=None, sim_md5=None):
        self.num_agents = num_agents
        self.dt = 1./fps
        self.fps = fps
        self.height = height
        self.width = width
        self.agent_radius = agent_radius
        self.centers = None
        self.velocities = None
        self.moving = None
        self.num_ticks = 0
        self.start_speed = None
        self.sim_duration = None

        self.social_state = Social_State(num_agents, **social_conf)
        self.desease_state = Desease_State(num_agents, **desease_conf)

        self.initialize_pos_vel()

    def initialize_pos_vel(self):
        x = (self.width - 4 * self.agent_radius) * np.random.random_sample((self.num_agents, 1)) + 2*self.agent_radius
        y = (self.height - 4 * self.agent_radius) * np.random.random_sample((self.num_agents, 1)) + 2*self.agent_radius

        self.centers = np.hstack((x, y)).astype(np.float64)

        diff = get_average_distance(self.centers)
        dist_to_achieve_spread = diff* 8 * self.desease_state.R_spread
        time_to_achieve_spread = self.desease_state.desease_duration

        self.start_speed = dist_to_achieve_spread/time_to_achieve_spread

        vx = np.random.random_sample((self.num_agents, 1)) - 0.5
        vy = np.random.random_sample((self.num_agents, 1)) - 0.5

        self.velocities = unify_speed(np.hstack((vx, vy)), self.start_speed)
        self.moving = np.ones(self.velocities.shape[0], dtype=bool)

    def draw_agents(self, screen):
        for anum, center in enumerate(self.centers[:,::-1].astype(np.int32)):
            pygame.draw.circle(screen, desease_color[self.desease_state.deseases[anum]], center, int(self.agent_radius))

    def display_stats(self):
        pass
        
    def is_finished(self):
        if self.desease_state.sim_sick == 0:
            #self.fig.show()
            return True
        return False

    def estimate_moving(self):
        #if self.num_ticks > self.social_onset * self.fps:
        self.moving = self.social_state.socials == social_state["active"]
        print("Moving: ", np.sum(self.moving))
        #else:
        #    self.moving = self.desease_state.deseases != desease_state["dead"]

    def update(self):
        self.num_ticks += 1
        self.sim_duration = self.num_ticks/self.fps
        self.estimate_moving()

        self.velocities[np.logical_not(self.moving), :] = 0.
        self.centers[self.moving] += self.velocities[self.moving]*self.dt

        inds1, inds2 = get_collisions(self.centers, r2=(2*self.agent_radius)**2, moving=self.moving)
        self.update_physics(self.moving, inds1, inds2)

        self.desease_state.update(self.dt, inds1, inds2)
        v_update, boost = self.social_state.update(self.desease_state, self.dt)

        if not (v_update is None):
            vx = np.random.random_sample((len(v_update), 1)) - 0.5
            vy = np.random.random_sample((len(v_update), 1)) - 0.5

            self.velocities[v_update] = unify_speed(np.hstack((vx, vy)), self.start_speed)*boost

        if self.num_ticks % self.fps == 0:
            self.social_state.print_stats(self.num_ticks/self.fps)
            self.desease_state.print_stats(self.num_ticks/self.fps)

    def print_final_report(self):
        #self.calc_KPIs()
        print("Scenario lasted: %.1f days" % self.sim_duration)
        print("Max sick people: %.2f" % self.desease_state.sim_sick_peak)
        print("People died: %.2f" % self.desease_state.sim_people_died)
        print("People never infected: %.2f" % self.desease_state.sim_people_infected)
        print("People working: %.2f" % self.social_state.sim_people_working)

    def calc_KPIs(self):
        d = self.desease_state
        #hist = {}
        #hist.update(d.history)
        #hist.update(self.social_state.history)
        #df = pd.DataFrame(hist)
        return dict(duration=self.sim_duration, sick_peak=d.sim_sick_peak, death=d.sim_people_died,
                    infected=d.sim_people_infected, working=self.social_state.sim_people_working,
                    start_speed=self.start_speed)

    def update_physics(self, moving, inds1, inds2):
        if len(inds1) > 0:
            self.velocities, self.centers = update_vectors(inds1, inds2,
                                                           self.velocities,
                                                           self.centers, (2*self.agent_radius)**2)

        for dim, max_dim in zip([0, 1], [self.width, self.height]):
            wall_hit = np.where(np.logical_or(np.logical_and(self.centers[:,dim] <= self.agent_radius,
                                                             self.velocities[:,dim] < 0),
                                              np.logical_and(self.centers[:,dim] >= (max_dim - self.agent_radius),
                                                             self.velocities[:,dim] > 0)
                                             ))[0]
            self.velocities[wall_hit, dim] = -self.velocities[wall_hit, dim]

def runPyGame(scenario, update_hook=None):
    # Initialise PyGame.
    """
    if visualize:
        pygame.init() 
        # Set up the clock. This will tick every frame and thus maintain a relatively constant framerate. Hopefully.
        fps = 60.0
        fpsClock = pygame.time.Clock()
        # Set up the window.
        width, height = scenario["width"], scenario["height"]
        print(width, height)
        screen = pygame.display.set_mode((height, width))
    """

    sim = Corona_Simulation(**scenario)
    # screen is the surface representing the window.
    # PyGame surfaces can be thought of as screen sections that you can draw onto.
    # You can also draw surfaces onto other surfaces, rotate surfaces, and transform surfaces.

    # Main game loop.
    while not sim.is_finished(): # Loop forever!
        sim.update()
        if not update_hook is None:
            update_hook(sim)
        #update(sim, visualize) # You can update/draw here, I've just moved the code for neatness.
        #if visualize:
        #    draw(screen, sim)
        #    dt = fpsClock.tick(fps)

    return sim.calc_KPIs()
