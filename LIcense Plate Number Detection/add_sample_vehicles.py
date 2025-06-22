# add_sample_vehicles.py

from db_connector import MongoDBConnector

def add_sample_vehicles():
    """
    Add sample vehicles to the database for testing
    """
    # Initialize MongoDB connector
    db = MongoDBConnector()
    
    try:
        # Add some sample vehicles
        vehicle1 = db.insert_vehicle(
            plate_number="AB123CD",
            status="stolen",
            owner={
                "name": "John Doe",
                "contact": "john.doe@example.com"
            },
            notes="Reported stolen on 2025-03-10"
        )
        
        vehicle2 = db.insert_vehicle(
            plate_number="XY987ZT",
            status="wanted",
            owner={
                "name": "Jane Smith",
                "contact": "jane.smith@example.com"
            },
            notes="Suspected involvement in fraud"
        )
        
        print(f"Added {2} sample vehicles to the database")
        
    except Exception as e:
        print(f"Error adding sample vehicles: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    add_sample_vehicles()