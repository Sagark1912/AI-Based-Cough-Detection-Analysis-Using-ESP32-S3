import ijson
import json
import sys
from pathlib import Path

src = Path('artifacts/features_timefreq_all_34434.complete.json')
report = Path('artifacts/features_timefreq_all_34434.verification.json')
count = 0
bad = []
seen = set()
first = None
last = None

try:
    with src.open('rb') as stream:
        for record in ijson.items(stream, 'item'):
            count += 1
            idx = int(record['row_index'])
            if first is None:
                first = idx
            last = idx
            if idx in seen:
                bad.append({'type': 'duplicate_row_index', 'row_index': idx})
            seen.add(idx)
            if idx != count - 1:
                bad.append({'type': 'unexpected_row_index', 'position': count - 1, 'row_index': idx})

            if record.get('ok') is True:
                timefreq = record.get('timefreq')
                complementary = record.get('complementary')
                if not isinstance(timefreq, list) or len(timefreq) != 128:
                    bad.append({'type': 'bad_timefreq_rows', 'row_index': idx})
                elif any(not isinstance(row, list) or len(row) != 128 for row in timefreq):
                    bad.append({'type': 'bad_timefreq_shape', 'row_index': idx})
                if not isinstance(complementary, list) or len(complementary) != 8:
                    bad.append({'type': 'bad_complementary_features', 'row_index': idx})
            elif not record.get('error'):
                bad.append({'type': 'failed_without_error', 'row_index': idx})

    result = {
        'file': str(src),
        'parsed_json': True,
        'records': count,
        'expected': 34434,
        'unique_row_indices': len(seen),
        'first_row_index': first,
        'last_row_index': last,
        'bad_record_count': len(bad),
        'complete': count == 34434 and len(seen) == 34434 and not bad,
        'errors': bad[:20],
    }
except Exception as exc:
    result = {
        'file': str(src),
        'parsed_json': False,
        'records': count,
        'expected': 34434,
        'complete': False,
        'error': f'{type(exc).__name__}: {exc}',
    }

report.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result, indent=2))
sys.exit(0 if result['complete'] else 2)
