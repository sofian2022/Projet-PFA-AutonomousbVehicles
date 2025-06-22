# db_connector.py

import os
import pymongo
import shutil
from datetime import datetime
import uuid

class MongoDBConnector:
    """
    Class to connect to MongoDB directly from Python using PyMongo
    """
    def __init__(self, connection_string="mongodb+srv://licensePlateRecg:password donnee par mongodb lors de la creation de cluster@cluster0.oqfvhel.mongodb.net/", 
                 db_name="license_plate_recognition", uploads_dir="./uploads"):
        """
        Initialize MongoDB connector
        
        Args:
            connection_string (str): MongoDB connection string
            db_name (str): Database name
            uploads_dir (str): Directory to store uploaded images
        """
        self.connection_string = connection_string
        self.db_name = db_name
        self.uploads_dir = uploads_dir
        self.client = None
        self.db = None
        
        # Create uploads directory if it doesn't exist
        os.makedirs(uploads_dir, exist_ok=True)
        
    def connect(self):
        """
        Connect to MongoDB
        """
        if not self.client:
            try:
                self.client = pymongo.MongoClient(self.connection_string)
                self.db = self.client[self.db_name]
                print("✅ MongoDB connected!")
            except Exception as e:
                print(f"❌ MongoDB connection error: {e}")
                raise
                
    def close(self):
        """
        Close MongoDB connection
        """
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            print("MongoDB connection closed")
            
    def save_detection(self, plate_number, coordinates, image_path, confidence, source='image', frame_number=None):
        """
        Save a license plate detection to MongoDB
        
        Args:
            plate_number (str): Detected license plate number
            coordinates (tuple): GPS coordinates (longitude, latitude)
            image_path (str): Path to the plate image
            confidence (float): Detection confidence
            source (str): Source of detection ('image' or 'video')
            frame_number (int, optional): Frame number for video detections
            
        Returns:
            dict: Vehicle document from MongoDB
        """
        try:
            # Ensure connection
            self.connect()
            
            # Create a timestamp for unique image naming
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            
            # Create a permanent path for the image in uploads directory
            file_extension = os.path.splitext(image_path)[1]
            filename = f"{plate_number}_{timestamp_str}_{uuid.uuid4().hex[:8]}{file_extension}"
            dest_path = os.path.join(self.uploads_dir, filename)
            
            # Copy the image file to uploads directory
            shutil.copy2(image_path, dest_path)
            
            # Create detection object
            detection = {
                "timestamp": timestamp,
                "location": {
                    "type": "Point",
                    "coordinates": coordinates  # [longitude, latitude]
                },
                "imageUrl": dest_path,
                "confidence": confidence,
                "source": source
            }
            
            if frame_number is not None:
                detection["frameNumber"] = frame_number
                
            # Find vehicle by plate number or create new one
            vehicles_collection = self.db["vehicles"]
            
            # Try to find existing vehicle
            vehicle = vehicles_collection.find_one({"plateNumber": plate_number})
            
            if vehicle:
                # Update existing vehicle
                vehicles_collection.update_one(
                    {"_id": vehicle["_id"]},
                    {"$push": {"detections": detection}}
                )
                # Get updated vehicle
                vehicle = vehicles_collection.find_one({"_id": vehicle["_id"]})
            else:
                # Create new vehicle
                new_vehicle = {
                    "plateNumber": plate_number,
                    "status": "normal",
                    "detections": [detection],
                    "createdAt": timestamp,
                    "updatedAt": timestamp
                }
                vehicles_collection.insert_one(new_vehicle)
                vehicle = new_vehicle
            
            print(f"✅ Detection saved for plate: {plate_number}")
            
            # Check if this is a vehicle of interest
            if vehicle.get("status") in ["stolen", "wanted"]:
                print(f"⚠️ ALERT: Vehicle with plate {plate_number} is marked as {vehicle['status']}!")
                # Here you could trigger additional logic like sending notifications
            
            return vehicle
            
        except Exception as e:
            print(f"❌ Error saving detection for plate {plate_number}: {e}")
            raise
            
    def check_vehicle_status(self, plate_number):
        """
        Check if a vehicle is in the database and get its status
        
        Args:
            plate_number (str): License plate number
            
        Returns:
            dict: Vehicle status information
        """
        try:
            # Ensure connection
            self.connect()
            
            vehicles_collection = self.db["vehicles"]
            vehicle = vehicles_collection.find_one({"plateNumber": plate_number})
            
            if not vehicle:
                return {"exists": False, "status": "unknown"}
            
            # Get the last detection if any
            last_detection = None
            if vehicle.get("detections") and len(vehicle["detections"]) > 0:
                last_detection = vehicle["detections"][-1]
            
            return {
                "exists": True,
                "status": vehicle.get("status", "normal"),
                "owner": vehicle.get("owner", None),
                "detectionCount": len(vehicle.get("detections", [])),
                "lastDetection": last_detection
            }
            
        except Exception as e:
            print(f"❌ Error checking vehicle status for plate {plate_number}: {e}")
            raise
            
    def insert_vehicle(self, plate_number, status="normal", owner=None, notes=None):
        """
        Insert a new vehicle into the database
        
        Args:
            plate_number (str): License plate number
            status (str): Vehicle status ('normal', 'stolen', 'wanted', 'expired')
            owner (dict): Owner information (name, contact)
            notes (str): Additional notes
            
        Returns:
            dict: Inserted vehicle document
        """
        try:
            # Ensure connection
            self.connect()
            
            vehicles_collection = self.db["vehicles"]
            
            # Check if vehicle already exists
            existing = vehicles_collection.find_one({"plateNumber": plate_number})
            if existing:
                print(f"Vehicle with plate {plate_number} already exists")
                return existing
            
            # Create new vehicle document
            timestamp = datetime.now()
            new_vehicle = {
                "plateNumber": plate_number,
                "status": status,
                "detections": [],
                "createdAt": timestamp,
                "updatedAt": timestamp
            }
            
            if owner:
                new_vehicle["owner"] = owner
                
            if notes:
                new_vehicle["notes"] = notes
                
            # Insert into database
            result = vehicles_collection.insert_one(new_vehicle)
            new_vehicle["_id"] = result.inserted_id
            
            print(f"✅ Vehicle with plate {plate_number} inserted")
            return new_vehicle
            
        except Exception as e:
            print(f"❌ Error inserting vehicle with plate {plate_number}: {e}")
            raise