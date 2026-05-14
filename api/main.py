import sys
import os

# Add wrapper to path so we can import MeridianManager
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from starlette.applications import Starlette
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from wrapper.meridian_manager import MeridianManager
import uvicorn
import json

manager = MeridianManager()

async def get_config(request):
    return JSONResponse(manager.get_config())

async def update_config(request):
    config = await request.json()
    manager.save_config(config)
    return JSONResponse({"status": "success", "config": manager.get_config()})

async def train_model(request):
    try:
        result = manager.train_model()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def get_data(request):
    try:
        df = manager.generate_data()
        # Use pandas to_json to handle numpy types and dates correctly
        json_data = df.to_json(orient="records", date_format="iso")
        return Response(content=json_data, media_type="application/json")
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def predict(request):
    try:
        spend_decisions = await request.json()
        result = manager.predict(spend_decisions)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

async def optimize(request):
    try:
        data = await request.json()
        total_budget = data.get("total_budget", 10000.0)
        fixed_allocations = data.get("fixed_allocations", {})
        result = manager.optimize_budget(total_budget, fixed_allocations)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

routes = [
    Route("/api/config", get_config, methods=["GET"]),
    Route("/api/config", update_config, methods=["POST"]),
    Route("/api/train", train_model, methods=["POST"]),
    Route("/api/data", get_data, methods=["GET"]),
    Route("/api/predict", predict, methods=["POST"]),
    Route("/api/optimize", optimize, methods=["POST"]),
]

app = Starlette(debug=True, routes=routes)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
