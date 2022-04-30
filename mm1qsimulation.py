import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from IPython.display import Markdown
from IPython.display import display
import warnings



warnings.filterwarnings("ignore", category=FutureWarning)

# settings for histogram plot
TITLE_SIZE = 22
HIST_BINS = 50

# generate a fixed seed of random numbers
rng = np.random.default_rng(seed=1)

# n = 1000  the number of jobs that is going to get inside the system
# lambda y  = mean_arrival_rate
# u -->>  mu = mean_service_rate
def runSimuationAndPlotResults(n, mean_arrival_rate, mean_service_rate):
        params = buildSimulationParameters(n, mean_arrival_rate, mean_service_rate)
        display(Markdown(
            f"### Simulation: n={n}, " +
            f"$\lambda$={format(params['mean_arrival_rate'])}, " +
            f"$\mu$={format(params['mean_service_rate'])}, " +
            f"$1/\lambda$={format(params['mean_interarrival_time'])}, " +
            f"$1/\mu$={format(params['mean_service_time'])}"
        ))
        #get the result for every single
        result = runSimulation(params)
        dumpStats(result)
        plotResults(result)

def buildSimulationParameters(num_jobs, mean_arrival_rate, mean_service_rate):
        return {
            "n": num_jobs,
            "mean_arrival_rate": mean_arrival_rate,
            "mean_service_rate": mean_service_rate,
            "mean_interarrival_time": 1.0 / mean_arrival_rate,
            "mean_service_time": 1.0 / mean_service_rate,
            "num_bins": int(num_jobs / mean_arrival_rate)
        }

def runSimulation(params):
        numberOfJobs = params["n"]

        # Parameters
        mean_interarrival_time = params["mean_interarrival_time"]
        mean_service_time = params["mean_service_time"]

        # Simulation data and results
        interarrival_times = rng.exponential(scale=mean_interarrival_time, size=numberOfJobs)
        arrival_times = np.cumsum(interarrival_times)
        service_times = rng.exponential(scale=mean_service_time, size=numberOfJobs)
        jobs_df = buildJobsDF(params, interarrival_times,
                              arrival_times, service_times)
        events_df = buildEventsDF(params, jobs_df)
        total_width = getTotalWidth(jobs_df)

        sim_mean_interarrival_time = jobs_df["interarrival_time"].mean()
        sim_mean_arrival_rate = 1.0 / sim_mean_interarrival_time
        sim_mean_service_time = jobs_df["service_time"].mean()
        sim_mean_service_rate = 1.0 / sim_mean_service_time
        sim_mean_wait_time = jobs_df["wait_time"].mean()
        sim_response_time_mean = jobs_df["response_time"].mean()
        sim_response_time_var = jobs_df["response_time"].var()

        # mean_num_jobs_in_system and mean_num_jobs_in_queue
        width = events_df["width"]
        total_weighted_num_jobs_in_system = (
                width * events_df["num_jobs_in_system"]).sum()
        total_weighted_num_jobs_in_queue = (
                width * events_df["num_jobs_in_queue"]).sum()
        sim_mean_num_jobs_in_system = total_weighted_num_jobs_in_system / total_width
        sim_mean_num_jobs_in_queue = total_weighted_num_jobs_in_queue / total_width

        # throughput mean and variance
        departures = events_df.loc[events_df["num_jobs_in_system_change"]
                                   == -1.0, "lo_bd"]
        hist, _ = np.histogram(departures, bins=int(total_width) + 1)
        sim_throughput_mean = np.mean(hist)

        # utilization
        util = estimateUtil(jobs_df)

        return {
            "params": params,
            "jobs_df": jobs_df,
            "events_df": events_df,
            "total_duration": total_width,
            "mean_arrival_rate": sim_mean_arrival_rate,
            "mean_interarrival_time": sim_mean_interarrival_time,
            "mean_service_rate": sim_mean_service_rate,
            "mean_service_time": sim_mean_service_time,
            "mean_wait_time": sim_mean_wait_time,
            "response_time_mean": sim_response_time_mean,
            "response_time_var": sim_response_time_var,
            "mean_num_jobs_in_system": sim_mean_num_jobs_in_system,
            "mean_num_jobs_in_queue": sim_mean_num_jobs_in_queue,
            "throughput_mean": sim_throughput_mean,
            "utilization": util,
        }


def buildJobsDF(params, interarrival_times, arrival_times, service_times):
        n = params["n"]

        jobs_df = pd.DataFrame({
            "interarrival_time": interarrival_times,
            "arrive_time": arrival_times,
            "service_time": service_times,
            "start_time": np.zeros(n),
            "depart_time": np.zeros(n)
        })

        jobs_df.loc[0, "start_time"] = jobs_df.loc[0, "arrive_time"]
        jobs_df.loc[0, "depart_time"] = jobs_df.loc[0,
                                                    "start_time"] + jobs_df.loc[0, "service_time"]

        for i in range(1, n):
            jobs_df.loc[i, "start_time"] = max(
                jobs_df.loc[i, "arrive_time"], jobs_df.loc[i - 1, "depart_time"])
            jobs_df.loc[i, "depart_time"] = jobs_df.loc[i,
                                                        "start_time"] + jobs_df.loc[i, "service_time"]

        jobs_df["response_time"] = jobs_df["depart_time"] - jobs_df["arrive_time"]
        jobs_df["wait_time"] = jobs_df["start_time"] - jobs_df["arrive_time"]

        return jobs_df


    # Serialize the jobs into events (arrival, start, departure) so we can compute job counts.

def buildEventsDF(params, jobs_df):
        n = params["n"]
        arrivals = jobs_df["arrive_time"]
        starts = jobs_df["start_time"]
        departures = jobs_df["depart_time"]

        # width = up_bd - lo_bd, num_jobs_in_queue = num_jobs_in_system - 1
        events_df = pd.DataFrame(
            columns=["lo_bd", "up_bd", "width", "num_jobs_in_system", "num_jobs_in_queue"])

        lo_bd = 0.0
        arrive_idx = 0
        start_idx = 0
        depart_idx = 0
        num_jobs_in_system = 0
        num_jobs_in_queue = 0

        while depart_idx < n:
            arrival = arrivals[arrive_idx] if arrive_idx < n else float("inf")
            start = starts[start_idx] if start_idx < n else float("inf")
            departure = departures[depart_idx]

            if arrival <= start and arrival <= departure:
                up_bd = arrival
                n_change, nq_change = 1, 1
                arrive_idx = arrive_idx + 1
            elif start <= arrival and start <= departure:
                up_bd = start
                n_change, nq_change = 0, -1
                start_idx = start_idx + 1
            else:
                up_bd = departure
                n_change, nq_change = -1, 0
                depart_idx = depart_idx + 1

            width = up_bd - lo_bd
            events_df = events_df.append({
                "lo_bd": lo_bd,
                "up_bd": up_bd,
                "width": width,
                "num_jobs_in_system": num_jobs_in_system,
                "num_jobs_in_queue": num_jobs_in_queue,
                "num_jobs_in_system_change": n_change,
                "num_jobs_in_queue_change": nq_change,
            }, ignore_index=True)

            num_jobs_in_system = num_jobs_in_system + n_change
            num_jobs_in_queue = num_jobs_in_queue + nq_change

            lo_bd = up_bd

        return events_df


def getTotalWidth(jobs_df):
        return jobs_df.iloc[-1]["depart_time"] - jobs_df.iloc[0]["arrive_time"]


def estimateUtil(jobs_df):
        busy = (jobs_df["depart_time"] - jobs_df["start_time"]).sum()
        return busy / getTotalWidth(jobs_df)

        # STATS DUMPS


def format(value):
        return f"{value:,.4f}"


def dumpStats(result):
        params = result["params"]
        jobs_df = result["jobs_df"]
        response_time = jobs_df["response_time"]
        arrival_rate_mean = result["mean_arrival_rate"]
        service_rate_mean = result["mean_service_rate"]
        service_time_mean = result["mean_service_time"]
        response_time_mean = result["response_time_mean"]
        throughput_mean = result["throughput_mean"]
        util = result["utilization"]
        num_jobs_in_system_mean = result["mean_num_jobs_in_system"]

        print("Simulation Results")
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        print("Total Duration of the system          = " + format(result['total_duration']))
        print("Arrival Rate Mean       = " + format(arrival_rate_mean))
        print("Interarrival Time Mean  = " +
              format(result['mean_interarrival_time']))
        print("Wait Time Mean          = " + format(result['mean_wait_time']))
        print("Service Rate Mean       = " + format(service_rate_mean))
        print("Service Time Mean       = " + format(service_time_mean))
        print("num_jobs_in_system_mean = " + format(num_jobs_in_system_mean))
        print("num_jobs_in_queue_mean  = " +
              format(result['mean_num_jobs_in_queue']))
        print("throughput_mean         = " + format(throughput_mean))
        print("Utilization of the system  = " + format(util))
        print("num_jobs_in_system_mean                = " +
              format(num_jobs_in_system_mean))
        print("")
        print("Arrival Rate Mean / Service Rate Mean   = " + format(arrival_rate_mean / service_rate_mean) +
              " (= " + format(arrival_rate_mean) + " / " + format(service_rate_mean) + ")")
        print("Utilization                            = " + format(util))
        print("Throughput Mean                        = " + format(throughput_mean))

# PLOT FUNCTIONS

def plotResults(result):
        params = result["params"]
        jobs_df = result["jobs_df"]
        events_df = result["events_df"]

        _plotJobsGantt(params, jobs_df)
        _plotJobsOverTime(events_df)
        _plotHistogram(params, jobs_df["interarrival_time"],
                        "Histogram of interarrival times", "Interarrival time")
        _plotHistogram(params, jobs_df["arrive_time"],
                        "Histogram of arrival times", "Arrival time")
        _plotHistogram(params, jobs_df["wait_time"],
                        "Histogram of wait times", "Wait time")
        _plotHistogram(params, jobs_df["service_time"],
                        "Histogram of service times", "Service time")
        _plotHistogram(params, jobs_df["response_time"],
                        "Histogram of response times", "Response time")


def _plotHistogram(params, data, title, xlabel):
        plt.figure(figsize=(14, 2))
        plt.title(title, size=TITLE_SIZE)
        plt.xlabel(xlabel)
        plt.ylabel("Count")
        plt.hist(data, bins=HIST_BINS)
        plt.show()


def _plotJobsGantt(params, jobs_df):
        n = params["n"]
        start_job = int(n / 2)
        end_job = start_job + 40
        trunc_df = jobs_df[start_job:end_job]

        plt.figure(figsize=(14, 8))
        plt.title("Job schedule (partial view)", size=TITLE_SIZE)
        plt.xlabel("Time")
        plt.ylabel("Job ID")
        plt.barh(
            y=trunc_df.index,
            left=trunc_df["arrive_time"],
            width=trunc_df["response_time"],
            alpha=1.0,
            color="gainsboro")
        plt.barh(
            y=trunc_df.index,
            left=trunc_df["start_time"],
            width=trunc_df["service_time"],
            alpha=1.0,
            color="limegreen")
        plt.gca().invert_yaxis()
        plt.grid(axis="x")
        plt.show()


    # FIXME Departures shouldn't count here.

def _plotJobsOverTime(events_df):
        plt.figure(figsize=(14, 2))
        plt.title("# jobs in system over time", size=TITLE_SIZE)
        plt.xlabel("Time")
        plt.ylabel("Job count")
        plt.plot(events_df["lo_bd"], events_df["num_jobs_in_system"])
        plt.show()

print("Welcome to MM1 Queue Simulation:")
print("The system will require the mean arrival rate and the mean service rate")
print("After the parameters the system will load to calculate the results")
print("System will give the output in console in addition to the histograms ")

print("--------------------------------------------")
numberOfJobsInSystem = input("Enter number Of Jobs In System: ")
print("--------------------------------------------")
meanArrivalRate = input("Enter mean_arrival_rate: ")
print("--------------------------------------------")
meanServiceRate = input("Enter mean_service_rate: ")

runSimuationAndPlotResults(int(numberOfJobsInSystem), int(meanArrivalRate), int(meanServiceRate))

x = input("press enter to exit")
