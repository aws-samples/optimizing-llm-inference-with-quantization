import plotly.graph_objects as go
from glob import glob
import argparse
import plotly
import orjson
import os


def create_chart(result_path: str, tasks: str, output_path: str):

    # loop over model names. for each, grab the in-context learning tasks and values and add a trace to the figure
    fig = go.Figure()

    tasks = tasks.replace(' ', '').split(",")
    values = []
    categories = []
    
    for filename in glob(os.path.join(result_path, "**","*.json")):
        model_name = filename.split("/")[-2]
        with open(filename, "rb") as f:
            results = orjson.loads(f.read())

        if tasks == "all":
            categories.extend(results["results"].keys())
        else:
            for k, v in results["results"].items():
                if k in tasks:
                    value = v.get("acc,none") or v.get("exact_match,get-answer")
                    values.append(value * 100.)
                    categories.append(k)

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            name=model_name
            ))

    fig.update_layout(
    polar=dict(
        radialaxis=dict(
        visible=True,
        range=[0., max(values)]
        )),
    showlegend=True
    )
    
    plotly.offline.plot(fig, filename=os.path.join(output_path, "chart_results.html"))
    fig.write_image(os.path.join(output_path, "chart_results.png")) 


def main():
    parser = argparse.ArgumentParser()

    # Path of evaluation results
    parser.add_argument("--result_path", type=str, default="./")

    # List of in-context tasks
    parser.add_argument("--tasks", type=str, default="all")

    # Path of chart
    parser.add_argument("--output_path", type=str, default="./")

    args = parser.parse_args()

    create_chart(**vars(args))


if __name__ == "__main__":
    main()