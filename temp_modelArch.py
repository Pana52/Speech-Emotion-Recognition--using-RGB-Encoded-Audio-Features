import pandas as pd
import tensorflow as tf
import io


def parse_summary(summary_string):
    # Initialize an empty list to store processed lines
    data = []
    lines = summary_string.split('\n')
    for line in lines:
        if line.startswith("Layer") or line.strip() == "":
            continue  # Skip the header or empty lines
        parts = line.split()
        if len(parts) >= 4:
            # Extract layer details assuming a standard summary format
            layer_type = ' '.join(parts[1:-2])  # Layer type might have spaces
            output_shape = parts[-2]  # Output shape
            param_num = parts[-1]  # Parameter number
            data.append([parts[0], layer_type, output_shape, param_num])
    # Create DataFrame if data is available
    if data:
        return pd.DataFrame(data, columns=["Layer", "Type", "Output Shape", "Parameters"])
    else:
        return pd.DataFrame(columns=["Layer", "Type", "Output Shape", "Parameters"])


def export_model_summaries_to_excel():
    models_to_load = {
        "DenseNet121": tf.keras.applications.DenseNet121,
        "EfficientNetB0": tf.keras.applications.EfficientNetB0,
        "ResNet50": tf.keras.applications.ResNet50,
        "VGG16": tf.keras.applications.VGG16
    }

    with pd.ExcelWriter('model_summaries.xlsx', engine='xlsxwriter') as writer:
        for name, model_func in models_to_load.items():
            buf = io.StringIO()
            model = model_func(include_top=True, weights=None)
            model.summary(print_fn=lambda x: buf.write(x + "\n"))
            summary_str = buf.getvalue()
            summary_df = parse_summary(summary_str)
            if not summary_df.empty:
                summary_df.to_excel(writer, sheet_name=name, index=False)


# Uncomment to execute in an environment where TensorFlow and Pandas are available
export_model_summaries_to_excel()
