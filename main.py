from modules.core import TrackingSystem

if __name__ == "__main__":

    app = TrackingSystem(config_path="config.json")
    
    # Chạy hệ thống
    app.run()