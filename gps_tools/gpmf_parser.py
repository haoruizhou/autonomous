import struct

CONTAINERS = {'DEVC', 'STRM', 'DVID'}

def unpack_value(type_char, data):
    """Unpack a single value based on GPMF type char."""
    if not data: return None
    
    try:
        if type_char == 'b': return struct.unpack('>b', data)[0]
        if type_char == 'B': return struct.unpack('>B', data)[0]
        if type_char == 'c': return struct.unpack('>c', data)[0] 
        if type_char == 'C': return struct.unpack('>B', data)[0]
        if type_char == 's': return struct.unpack('>h', data)[0]
        if type_char == 'S': return struct.unpack('>H', data)[0]
        if type_char == 'l': return struct.unpack('>i', data)[0]
        if type_char == 'L': return struct.unpack('>I', data)[0]
        if type_char == 'f': return struct.unpack('>f', data)[0]
        if type_char == 'd': return struct.unpack('>d', data)[0]
        if type_char == 'J': return struct.unpack('>Q', data)[0]
        
        # Raw types or unknown
        return data
    except Exception:
        return data

def parse_gpmf(data, parent_path=None):
    if parent_path is None:
        parent_path = []
        
    idx = 0
    length = len(data)
    
    while idx + 8 <= length:
        try:
            key = data[idx:idx+4].decode('utf-8', errors='ignore')
        except:
            key = "UNKN"
            
        type_byte = data[idx+4]
        type_char = chr(type_byte) if type_byte != 0 else '\x00'
        structure_size = data[idx+5]
        
        try:
            repeat = struct.unpack('>H', data[idx+6:idx+8])[0]
        except:
            repeat = 0
            
        idx += 8
        total_size = structure_size * repeat
        
        # Alignment
        mod = total_size % 4
        padding = (4 - mod) % 4
        aligned_total = total_size + padding
        
        if idx + total_size > length:
            break
            
        payload = data[idx:idx+total_size]
        idx += aligned_total
        
        is_container = key in CONTAINERS
        
        if is_container:
            new_path = parent_path + [key]
            yield (parent_path, key, None)
            yield from parse_gpmf(payload, new_path)
            
        else:
            final_values = []
            
            # Determine element size
            type_size = 0
            if type_char in 'sS': type_size = 2
            elif type_char in 'lLfF': type_size = 4
            elif type_char in 'dDJ': type_size = 8
            elif type_char in 'bBcHC': type_size = 1
            
            # If type_size is 0 (Unknown), treat as raw blob per repeat
            if type_size == 0:
                 for i in range(repeat):
                    start = i * structure_size
                    chunk = payload[start : start+structure_size]
                    final_values.append(chunk)
            else:
                # Known type parsing
                elements_per_item = 1
                if structure_size > type_size:
                    elements_per_item = structure_size // type_size
                
                # Verify structure_size is valid for type
                if structure_size >= type_size and (structure_size % type_size) == 0:
                    for i in range(repeat):
                        item_offset = i * structure_size
                        item_chunk = payload[item_offset : item_offset + structure_size]
                        
                        if elements_per_item > 1:
                            row = []
                            for k in range(elements_per_item):
                                sub = item_chunk[k*type_size : (k+1)*type_size]
                                val = unpack_value(type_char, sub)
                                row.append(val)
                            final_values.append(row)
                        else:
                            val = unpack_value(type_char, item_chunk)
                            final_values.append(val)
                else:
                    # Mismatch of size and type, just yield chunks
                     for i in range(repeat):
                        start = i * structure_size
                        chunk = payload[start : start+structure_size]
                        final_values.append(chunk)

            yield (parent_path, key, final_values)
