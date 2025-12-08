from framework import AdaptiveMonitoringFramework

def main():
    # Dummy values for init; base_path/csv_files not used if we only load model
    framework = AdaptiveMonitoringFramework(base_path="", csv_files=[])

    framework.load_state("saved_models/phase2_adapted", version_name="mon_thu_plus_fri")

    print("[Test] Loaded model + preprocessing successfully.")

if __name__ == "__main__":
    main()