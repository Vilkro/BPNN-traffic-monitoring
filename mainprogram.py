from framework import AdaptiveMonitoringFramework

base_path = r"D:/TrafficClassification/MachineLearningCVE"

csv_files = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
]

def main():
    framework = AdaptiveMonitoringFramework(base_path, csv_files)

    print("\n=== Loading and preparing data ===")
    framework.load_and_prepare()

    print("\n=== Phase 1: Baseline (Mon–Thu -> Friday) ===")
    metrics_p1, packed = framework.run_phase1_baseline(
        n_per_class_mon_thu=200000,    # adjusted
        n_per_class_fri=200000
    )

    framework.save_current_state("saved_models/phase1_mon_thu", version_name="mon_thu_only")

    print("\n=== Phase 2: Adaptation (Mon–Thu + Friday-train -> Friday-test) ===")
    metrics_p2 = framework.run_phase2_adaptation(*packed)

    framework.save_current_state("saved_models/phase2_adapted", version_name="mon_thu_plus_fri")

    print("\n=== Summary ===")
    print("Phase 1 (baseline) accuracy:", metrics_p1["accuracy"])
    print("Phase 2 (adapted)  accuracy:", metrics_p2["accuracy"])

if __name__ == "__main__":
    main()

