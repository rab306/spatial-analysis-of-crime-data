# The following function is to solve longitudional levelling in surveying works
# I used this function to solve multiple levelings in my graduation project where the observation were stored in different sheets in excel file

import pandas as pd

def compute_reduced_levels(file_path, output_file_path, benchmarks):
    xls = pd.ExcelFile(file_path)
    
    # Iterate over each sheet in the Excel file
    with pd.ExcelWriter(output_file_path) as writer:       
        for sheet_name in xls.sheet_names:
            data = pd.read_excel(file_path, sheet_name=sheet_name)
            
            benchmark = benchmarks.get(sheet_name, None)
            if benchmark is None:
                print(f"No benchmark found for sheet '{sheet_name}'. Skipping...")
                continue
            
            reduced_levels = [benchmark]  
            instrument_heights = []

            bs_readings = data['BS']
            fs_readings = data['FS']
            is_readings = data.get('IS')  # Use .get() to handle cases where 'IS' column may not exist

            for i in range(len(data)):
                if pd.notna(fs_readings[i]):
                    reduced_level = instrument_heights[i-1] - fs_readings[i]
                    reduced_levels.append(reduced_level)
                elif is_readings is not None and pd.notna(is_readings[i]):
                    reduced_level = instrument_heights[i-1] - is_readings[i]
                    reduced_levels.append(reduced_level)

                # Compute instrument height
                if pd.notna(bs_readings[i]):
                    instrument_height = reduced_levels[-1] + bs_readings[i]
                else:
                    instrument_height = instrument_heights[-1]  # Keep instrument height the same if BS reading is not available

                instrument_heights.append(instrument_height)

            instrument_heights[-1] = None

            data['Reduced Level'] = reduced_levels  # Exclude the benchmark from reduced levels
            data['Instrument Height'] = instrument_heights

            data.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Levelling computation completed", output_file_path)
    return "Levelling computation completed"
