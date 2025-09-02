"""
Enhanced Sales Big Data Analysis Pipeline
Integrates advanced ML algorithms, feature engineering, and ensemble methods
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['LOKY_MAX_CPU_COUNT'] = '4'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Add script directory to path
sys.path.append('script')

# Import enhanced modules
from script.enhanced_main_pipeline import run_enhanced_pipeline
from script.advanced_visualizations import AdvancedVisualizer
import matplotlib.pyplot as plt

def main():
    """Main execution function"""
    print("🚀 Starting Enhanced Sales Big Data Analysis Pipeline")
    print("=" * 60)

    # Ask user for mode
    mode = input("Select mode: [R]esearch or [D]emo: ").strip().lower()
    if mode not in ["r", "d"]:
        print("❌ Invalid choice. Defaulting to Demo Mode.")
        mode = "d"

    is_demo = (mode == "d")
    print(f"\n⚡ Running in {'DEMO' if is_demo else 'RESEARCH'} mode\n")

    # Initialize visualizer
    visualizer = AdvancedVisualizer()
    
    try:
        # Run the enhanced pipeline
        pipeline, final_report = run_enhanced_pipeline(demo_mode=is_demo)
        
        if pipeline is None:
            print("❌ Pipeline failed to complete")
            return
        
        # Save results
        results_dir = "results/demo" if is_demo else "results/research"
        os.makedirs(results_dir, exist_ok=True)
        final_report.to_csv(os.path.join(results_dir, "final_report.csv"), index=False)
        print(f"✅ Results saved to {results_dir}")
        
        # Create additional advanced visualizations
        print("\n🎨 Creating advanced visualizations...")
        
        # Model performance radar chart
        if final_report is not None and len(final_report) > 0:
            visualizer.plot_model_performance_radar(final_report)
        
        # Create interactive dashboard
        if os.path.exists('results/final_processed_dataset.csv'):
            df = pd.read_csv('results/final_processed_dataset.csv')
            visualizer.create_interactive_dashboard(df, pipeline.results_summary)
        
        print("\n✅ Enhanced pipeline completed successfully!")
        print("\n📁 Results saved in:")
        print("  - results/comprehensive_model_report.csv")
        print("  - results/interactive_dashboard.html")
        print("  - results/model_performance_radar.html")
        print("  - models/ (trained models)")
        print("  - results/ (all visualizations and metrics)")
        
        # Display final summary
        if final_report is not None:
            print("\n📊 Top Performing Models:")
            print("-" * 40)
            
            # Group by task and show best model
            if 'Task' in final_report.columns:
                for task in final_report['Task'].unique():
                    task_results = final_report[final_report['Task'] == task]
                    
                    if 'R²' in task_results.columns:
                        best_model = task_results.loc[task_results['R²'].idxmax()]
                        print(f"{task}: {best_model['Model']} (R² = {best_model['R²']:.4f})")
                    elif 'F1' in task_results.columns:
                        best_model = task_results.loc[task_results['F1'].idxmax()]
                        print(f"{task}: {best_model['Model']} (F1 = {best_model['F1']:.4f})")
                    elif 'Silhouette Score' in task_results.columns:
                        best_model = task_results.loc[task_results['Silhouette Score'].idxmax()]
                        print(f"{task}: {best_model['Model']} (Silhouette = {best_model['Silhouette Score']:.4f})")
            else:
                print("No 'Task' column found in final_report, skipping task loop.")
        
    except Exception as e:
        print(f"❌ Error in main execution: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Show any pending plots
        plt.show()

if __name__ == "__main__":
    main()