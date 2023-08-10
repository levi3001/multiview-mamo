from models_multiview import multiview_detector, multiview_detector_concat, multiview_detector_swint
create_model ={'multiview_detector': multiview_detector.create_model,
               'multiview_detector_concat': multiview_detector_concat.create_model,
               'multiview_detector_swint': multiview_detector_swint.create_model}