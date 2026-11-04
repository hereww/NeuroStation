"""Event association evidence; software and device clocks are never equated."""
from collections import defaultdict
import math
import numpy as np


def associate_events(events, markers, timestamps):
    groups=defaultdict(list)
    result=[]
    for event in events:
        row={'event_id':event.get('event_id'), 'trial_index':event.get('trial_index'),
             'event_name':event.get('event_name'), 'marker_code':event.get('marker_code'),
             'sample_index':None, 'device_timestamp_s':None, 'association':'unknown',
             'physical_stimulus_onset_s':None, 'timestamp_source':'unknown',
             'software_request_time_s':event.get('event_monotonic') if event.get('event_monotonic') is not None else event.get('monotonic_s'),
             'software_clock':event.get('software_clock','legacy_host_monotonic_unspecified_basis'),
             'display_callback_time_s':event.get('display_callback_time_s'),
             'marker_request_time_s':event.get('marker_request_time_s'),
             'marker_api_return_time_s':event.get('marker_api_return_time_s'),
             'marker_acknowledgement':'host_API_return_only' if event.get('marker_api_return_time_s') is not None else 'unknown',
             'stimulus_acknowledged':event.get('stimulus_acknowledged',None)}
        result.append(row)
        try:
            code=float(event.get('marker_code'))
            if math.isfinite(code) and code != 0:
                groups[code].append(row)
        except (TypeError,ValueError):
            pass
    for code, rows in groups.items():
        matches=np.flatnonzero(np.asarray(markers)==code)
        if len(matches)!=len(rows):
            for row in rows: row['reason']='missing_or_ambiguous_marker_count'
            continue
        for row,index in zip(rows,matches):
            row.update(sample_index=int(index),association='inferred_marker_order',
                       timestamp_source='brainflow_device_unverified')
            if timestamps is not None and index<len(timestamps) and np.isfinite(timestamps[index]):
                row['device_timestamp_s']=float(timestamps[index])
    return {'schema_version':2,'interval':'zero_based_half_open','events':result,
            'limitations':['Marker order is not photodiode-confirmed stimulus onset.',
                           'Device and host clock offset/drift have not been calibrated.']}
