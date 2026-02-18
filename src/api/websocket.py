import logging
from typing import Optional

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        logger.info(f"WebSocket connected for job: {job_id}")

    def disconnect(self, websocket: WebSocket, job_id: str):
        if job_id in self.active_connections:
            if websocket in self.active_connections[job_id]:
                self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        logger.info(f"WebSocket disconnected for job: {job_id}")

    async def send_progress(self, job_id: str, progress: int, message: str, stage: str = ""):
        if job_id in self.active_connections:
            data = {
                "type": "progress",
                "job_id": job_id,
                "progress": progress,
                "message": message,
                "stage": stage
            }
            disconnected = []
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending to WebSocket: {e}")
                    disconnected.append(connection)

            for conn in disconnected:
                self.disconnect(conn, job_id)

    async def send_completion(self, job_id: str, success: bool, message: str, result_path: Optional[str] = None):
        if job_id in self.active_connections:
            data = {
                "type": "completed",
                "job_id": job_id,
                "success": success,
                "message": message,
                "result_path": result_path
            }
            for connection in self.active_connections[job_id]:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logger.error(f"Error sending completion: {e}")

    async def broadcast(self, message: dict):
        for _job_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error(f"Error broadcasting: {e}")


manager = ConnectionManager()
