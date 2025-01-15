from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import math
import time
import cv2
import numpy as np
import requests
import threading
import torch
import json
import os
# Default constants
DEFAULT_YELLOW = 5
DEFAULT_GREEN = 15
DEFAULT_MINIMUM = 10
DEFAULT_MAXIMUM = 60
DEFAULT_RED = 130
BASE_URL = os.getenv("BASE_URL", "https://google.com")
VEHICLE_TIMES = {
    "cars": 2,
    "bikes": 1,
    "buses": 3,
    "trucks": 4,
    "rickshaws": 2,
}

VIDEO_FEEDS = []  # Signal video feeds
SIGNAL_ID = []  # Signal IDs

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
CLASSES = model.names

app = FastAPI()


def get_api_data(api_url):
    try:
        response = requests.get(api_url)
        if response.status_code != 200:
            print(f"Failed to fetch data from API. Status code: {response.status_code}")
            return None

        data = response.json()
        # Extract video feeds and signal IDs
        global VIDEO_FEEDS, SIGNAL_ID
        VIDEO_FEEDS = [signal["signal_image"] for signal in data["signals"]]
        SIGNAL_ID = [signal["signal_id"] for signal in data["signals"]]

        print(f"VIDEO_FEEDS updated with URLs: {VIDEO_FEEDS}")
        return data
    except Exception as e:
        print(f"Error fetching API data: {e}")
        return None


def detect_vehicles(image_url):
    try:
        img_arr = np.asarray(bytearray(requests.get(image_url).content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)
        if img is None:
            print(f"Error: Unable to load image from {image_url}")
            return {key: 0 for key in VEHICLE_TIMES.keys()}

        img = cv2.resize(img, (640, 640))
        results = model(img)

        vehicle_counts = {key: 0 for key in VEHICLE_TIMES.keys()}
        for *xyxy, conf, cls in results.xyxy[0]:  # Results in xyxy format (x1, y1, x2, y2)
            label = CLASSES[int(cls)]
            if label == "car":
                vehicle_counts["cars"] += 1
            elif label == "bus":
                vehicle_counts["buses"] += 1
            elif label == "truck":
                vehicle_counts["trucks"] += 1
            elif label == "bike":
                vehicle_counts["bikes"] += 1

        return vehicle_counts
    except Exception as e:
        print(f"Error detecting vehicles: {e}")
        return {key: 0 for key in VEHICLE_TIMES.keys()}


def calculate_green_time(image_index):
    vehicle_counts = detect_vehicles(VIDEO_FEEDS[image_index])
    print(f"Vehicle counts for Signal {image_index + 1}: {vehicle_counts}")
    total_vehicle_count = sum(vehicle_counts.values())
    green_time = sum(vehicle_counts[vehicle] * VEHICLE_TIMES[vehicle] for vehicle in vehicle_counts)
    return max(DEFAULT_MINIMUM, min(math.ceil(green_time / 2), DEFAULT_MAXIMUM)), total_vehicle_count


def send_to_api(api_url, signal_data):
    try:
        response = requests.patch(api_url, json={"signals": signal_data})
        if response.status_code == 200:
            print("Successfully updated signals")
        else:
            print(f"Failed to update signals: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")


@app.get("/initialize")
def initialize(id: str = Query(...)):
    """
    Initialize the video feeds and signal data based on the given ID.
    """
    api_url = f"{BASE_URL}/v1/signal/{id}"
    data = get_api_data(api_url)
    if not data:
        raise HTTPException(status_code=500, detail="Failed to fetch data from API")
    return {"message": "Initialization successful", "data": data}


@app.get("/control")
def control_traffic(id: str = Query(...)):
    """
    Control traffic signals dynamically for the given ID.
    """
    global SIGNAL_ID, VIDEO_FEEDS

    # Ensure the system is initialized
    if not SIGNAL_ID or not VIDEO_FEEDS:
        raise HTTPException(status_code=400, detail="System not initialized. Call /initialize first.")

    api_url = f"{BASE_URL}/v1/signal/{id}/update-count"
    signals_count = len(VIDEO_FEEDS)
    current_signal = 0

    red_duration_tracker = {i: DEFAULT_RED for i in range(signals_count)}
    signal_data = []

    # Process each signal
    for _ in range(signals_count):
        green_time_current_signal, total_vehicle_count = calculate_green_time(current_signal)

        signal_data.append({
            "signal_id": SIGNAL_ID[current_signal],
            "red_duration": 0,
            "yellow_duration": DEFAULT_YELLOW,
            "green_duration": green_time_current_signal,
            "vehicle_count": total_vehicle_count
        })

        # Update other signals
        for i in range(signals_count):
            if i != current_signal:
                red_time = DEFAULT_RED if i != (current_signal + 1) % signals_count else green_time_current_signal + DEFAULT_YELLOW
                signal_data.append({
                    "signal_id": SIGNAL_ID[i],
                    "red_duration": red_time,
                    "yellow_duration": DEFAULT_YELLOW,
                    "green_duration": DEFAULT_GREEN
                })

        send_to_api(api_url, signal_data)
        current_signal = (current_signal + 1) % signals_count

    return {"message": "Traffic control completed successfully", "data": signal_data}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
