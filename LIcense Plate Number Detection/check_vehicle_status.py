 

from db_connector import MongoDBConnector
import sys
import yagmail

# ====== CONFIGURATION ======
POLICE_EMAIL = "soufiane0elyaa-----@gmail.com"  # Replace with actual police email
SENDER_EMAIL = "elyaakoubisouf-----@gmail.com"       # Use a Gmail or SMTP-capable email
APP_PASSWORD = "vous pouvez le trouver sur votre compte gmail"        # Gmail App Password or SMTP password

def send_alert_email(plate_number, coordinates, image_url, status):
    subject = f"ALERT: {status.upper()} Vehicle Detected - {plate_number}"

    # Styling using HTML
    body = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: Arial, sans-serif;
                color: #333;
                margin: 0;
                padding: 0;
                line-height: 1.4;  /* Adjusted line height for normal paragraph spacing */
            }}
            h1 {{
                color: #D32F2F;
                font-size: 22px;
                margin-bottom: 10px;
            }}
            p {{
                font-size: 14px;
                margin-bottom: 10px;
            }}
            .alert {{
                background-color: #FFEBEE;
                padding: 10px;
                border-left: 5px solid #D32F2F;
                margin-bottom: 10px;
            }}
            .location {{
                font-weight: bold;
            }}
            .image {{
                margin-top: 15px;
                margin-bottom: 15px;
            }}
            .image img {{
                width: 100%;  /* Scale image properly */
                max-width: 600px;  /* Maximum width */
            }}
        </style>
    </head>
    <body>
        <h1>ALERT: {status.upper()} Vehicle Detected</h1>
        <p><strong>Plate Number:</strong> {plate_number}</p>
        <div class="alert">
            <p><strong>Status:</strong> {status.upper()}</p>
            <p><strong>Location:</strong> Longitude {coordinates[0]}, Latitude {coordinates[1]}</p>
            <p><strong>Detection Image:</strong></p>
            <div class="image">
                <img src="{image_url}" alt="Vehicle Image"/>
            </div>
        </div>
        <p>Please take immediate action.</p>
        <footer>
            <p>- Automated Vehicle Monitoring System</p>
        </footer>
    </body>
    </html>
    """
    
    try:
        yag = yagmail.SMTP(SENDER_EMAIL, APP_PASSWORD)
        yag.send(
            to=POLICE_EMAIL,
            subject=subject,
            contents=[body]  # Ensure this is in HTML format
        )
        print("?? Alert email sent to police.")
    except Exception as e:
        print(f"? Failed to send email: {e}")

def check_vehicle_status(plate_number):
    """
    Check status of a vehicle by plate number
    """
    db = MongoDBConnector()
    
    try:
        status_info = db.check_vehicle_status(plate_number)
        
        if status_info["exists"]:
            print(f"? Vehicle found: {plate_number}")
            print(f"Status: {status_info['status']}")

            if status_info["owner"]:
                print(f"Owner: {status_info['owner']['name']}")
                print(f"Contact: {status_info['owner']['contact']}")
            
            print(f"Detection count: {status_info['detectionCount']}")

            if status_info["lastDetection"]:
                last_detection = status_info["lastDetection"]
                timestamp = last_detection['timestamp']
                coords = last_detection['location']['coordinates']
                image_url = last_detection['imageUrl']

                print(f"Last detected: {timestamp}")
                print(f"Location: {coords}")
                print(f"Image: {image_url}")

                if status_info["status"] in ["stolen", "wanted"]:
                    send_alert_email(
                        plate_number=plate_number,
                        coordinates=coords,
                        image_url=image_url,
                        status=status_info["status"]
                    )
        else:
            print(f"? Vehicle not found: {plate_number}")

    except Exception as e:
        print(f"Error checking vehicle status: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        check_vehicle_status(sys.argv[1])
    else:
        print("Please provide a license plate number")
        print("Example: python check_vehicle_status.py AB123CD")

