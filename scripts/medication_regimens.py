"""Source-aligned medication names, doses, routes, and frequencies."""
import json
import re

from scripts import medication_doses, name_spans, token_classifier

VERSION = 'medication-regimen-bio-v1'
PROTOCOL = 'medication-route-frequency-candidates-v1'
CANDIDATE_VERSION = 'route-frequency-candidates-v1'
LINKING_RULE = 'local-regimen-links-v1'
LABELS = medication_doses.LABELS + ['B-ROUTE','I-ROUTE','B-FREQUENCY','I-FREQUENCY']
FIELDS = {'route':5,'frequency':7}
_ROUTE = re.compile(r'(?<!\w)(?:by mouth|per os|oral(?:ly)?|intravenous(?:ly)?|i\.v\.|IV|p\.o\.|PO|'
                    r'intramuscular(?:ly)?|i\.m\.|IM|subcutaneous(?:ly)?|s\.c\.|SC|SQ|'
                    r'intrathecal(?:ly)?|epidural(?:ly)?|nebulized|inhaled|inhalation|topical(?:ly)?|'
                    r'transdermal(?:ly)?|rectal(?:ly)?|sublingual(?:ly)?|SL|intranasal(?:ly)?)(?!\w)',re.I)
_COUNT = r'(?:\d+(?:\s*[-–]\s*\d+)?|one|two|three|four|six|eight|twelve)'
_PERIOD = r'(?:hours?|hrs?|h|days?|weeks?|months?)'
_FREQUENCY = re.compile(r'(?<!\w)(?:(?:once|twice|thrice|'+_COUNT+r'\s+times)\s+(?:daily|weekly|monthly|'
                       r'(?:a|per)\s+(?:day|week|month))|every\s+other\s+day|on\s+alternate\s+days|'
                       r'every\s+'+_COUNT+r'\s*'+_PERIOD+r'|(?:every|each)\s+(?:morning|evening|night)|'
                       r'at\s+bedtime|with\s+meals|q\s*\d+\s*h|b\.i\.d\.|t\.i\.d\.|q\.i\.d\.|'
                       r'q\.d\.|p\.r\.n\.|BID|TID|QID|QD|QOD|PRN|daily|weekly|monthly|nightly|as\s+needed)'
                       r'(?:\s+as\s+needed)?(?!\w)',re.I)
_BEFORE_NAME = re.compile(r'(?:\s|<(?:DOSE|ROUTE|FREQUENCY)>|\b(?:of|by|via)\b)*',re.I)
_BRIDGE = re.compile(r'(?:\s|[,:(=]|<(?:DOSE|ROUTE|FREQUENCY)>|\b(?:at|a|the|dose|dosage|of|in|with|via|by|'
                     r'was|is|given|administered|prescribed|take|taking|to|and)\b)*',re.I)


def candidates(record):
    text=record['text']
    tokens=name_spans.source_tokens(text)
    starts={t['start'] for t in tokens};ends={t['end'] for t in tokens}
    names=name_spans.resolve_response(record,json.dumps(name_spans.reference_payload(record)))
    occupied=names+record['dose_mentions']
    result={}
    for field,pattern in [('route',_ROUTE),('frequency',_FREQUENCY)]:
        result[field]=[{'start':m.start(),'end':m.end(),'text':m.group()} for m in pattern.finditer(text)
                       if m.start() in starts and m.end() in ends
                       and not any(m.start()<s['end'] and s['start']<m.end() for s in occupied)]
    return result


def marked_source(record):
    names=name_spans.resolve_response(record,json.dumps(name_spans.reference_payload(record)))
    amounts=record['dose_mentions'];options=candidates(record)
    tagged=[(s['start'],s['end'],f'{kind}_{i}') for kind,spans in [('MED',names),('DOSE',amounts),
            ('ROUTE',options['route']),('FREQUENCY',options['frequency'])] for i,s in enumerate(spans)]
    pieces=[];cursor=0
    for start,end,tag in sorted(tagged):
        if start<cursor:raise ValueError('Overlapping marked regimen entities')
        pieces.extend([record['text'][cursor:start],f'⟦{tag}:',record['text'][start:end],'⟧']);cursor=end
    pieces.append(record['text'][cursor:])
    return ''.join(pieces),names,options


def validate_response(record,response):
    value=json.loads(response)
    names=name_spans.reference_payload(record)['medication_name_spans']
    if (not isinstance(value,dict) or set(value)!={'reviewed_medication_indices','routes','frequencies','coverage_reviewed'}
            or value['coverage_reviewed'] is not True or not isinstance(value['reviewed_medication_indices'],list)
            or any(type(i) is not int for i in value['reviewed_medication_indices'])
            or sorted(value['reviewed_medication_indices'])!=list(range(len(names)))):
        raise ValueError('Every frozen medication occurrence needs explicit route/frequency review')
    result=dict(record,regimen_label_protocol=PROTOCOL)
    options=candidates(record)
    for field,plural in [('route','routes'),('frequency','frequencies')]:
        items=value[plural]
        if not isinstance(items,list):raise ValueError('Attribute selections must be arrays')
        if not names and items:raise ValueError('Attributes require a frozen medication mention')
        selected=set();links=set();seen=set()
        for item in items:
            if not isinstance(item,dict) or set(item)!={'medication_index',f'{field}_index'}:
                raise ValueError('Attribute selection has unexpected fields')
            med,index=item['medication_index'],item[f'{field}_index']
            if type(index) is not int or not 0<=index<len(options[field]):
                raise ValueError('Unknown source attribute candidate')
            if med is not None and (type(med) is not int or not 0<=med<len(names)):
                raise ValueError('Unknown medication occurrence')
            if (med,index) in seen:raise ValueError('Duplicate attribute selection')
            seen.add((med,index));selected.add(index)
            if med is not None:links.add((med,index))
        indices=sorted(selected)
        result[f'{field}_mentions']=[options[field][i] for i in indices]
        result[f'{field}_relations']=[{'medication_index':m,f'{field}_index':indices.index(i)} for m,i in sorted(links)]
    reference_spans(result)
    return result


def reference_spans(record):
    if record.get('regimen_label_protocol')!=PROTOCOL:
        raise ValueError('Missing or incompatible regimen protocol')
    dose_pairs=medication_doses.reference_spans(record)
    words=name_spans.source_tokens(record['text'])
    starts={t['start']:i+1 for i,t in enumerate(words)};ends={t['end']:i+1 for i,t in enumerate(words)}
    name_pairs=name_spans.reference_payload(record)['medication_name_spans']
    occupied={i for a,b in name_pairs+dose_pairs for i in range(a,b+1)}
    result={}
    for field in FIELDS:
        pairs=[];previous=-1
        for span in record[f'{field}_mentions']:
            a,b=span['start'],span['end']
            if (type(a) is not int or type(b) is not int or a not in starts or b not in ends
                    or a<previous or a>=b or record['text'][a:b]!=span['text']):
                raise ValueError('Attribute spans must be sorted, disjoint, exact, and token aligned')
            pair=[starts[a],ends[b]];covered=set(range(pair[0],pair[1]+1))
            if occupied&covered:raise ValueError('Regimen entity spans overlap')
            occupied.update(covered);pairs.append(pair);previous=b
        seen=set()
        for link in record[f'{field}_relations']:
            if not isinstance(link,dict) or set(link)!={'medication_index',f'{field}_index'}:
                raise ValueError('Invalid attribute relation schema')
            med,index=link['medication_index'],link[f'{field}_index']
            if (type(med) is not int or type(index) is not int or not 0<=med<len(name_pairs)
                    or not 0<=index<len(pairs) or (med,index) in seen):
                raise ValueError('Invalid attribute relation')
            seen.add((med,index))
        result[field]=pairs
    return result


def word_labels(record):
    labels=medication_doses.word_labels(record)
    for field,pairs in reference_spans(record).items():
        for start,end in pairs:
            labels[start-1]=FIELDS[field];labels[start:end]=[FIELDS[field]+1]*(end-start)
    return labels


def encode_record(record,tokenizer,max_length=512,labeled=True):
    return token_classifier.encode_record(record,tokenizer,max_length,labeled,label_function=word_labels)


def decode_predictions(word_ids,predictions,word_count):
    if any(type(p) is not int or not 0<=p<len(LABELS) for p in predictions):
        raise ValueError('Unknown regimen BIO label')
    result={}
    for key,start in [('medication_name_spans',1),('dose_spans',3),('route_spans',5),('frequency_spans',7)]:
        selected=[p-start+1 if p in (start,start+1) else 0 for p in predictions]
        result[key]=token_classifier.decode_predictions(word_ids,selected,word_count)['medication_name_spans']
    return result


def attribute_spans(text,payload,field):
    return name_spans.resolve_response({'text':text},json.dumps({'medication_name_spans':payload[f'{field}_spans']}))


def candidate_payload(text,base):
    names,doses=medication_doses.resolve(text,base)
    source={'text':text,'name_mentions':names,'name_label_protocol':name_spans.NAME_PROTOCOL,'dose_mentions':doses}
    options=candidates(source)
    words=name_spans.source_tokens(text)
    starts={word['start']:i+1 for i,word in enumerate(words)};ends={word['end']:i+1 for i,word in enumerate(words)}
    return dict(base,**{f'{field}_spans':[[starts[s['start']],ends[s['end']]] for s in spans] if names else []
                       for field,spans in options.items()})


def validated_payload(text,payload):
    names,doses=medication_doses.resolve(text,payload)
    source={'text':text,'name_mentions':names,'name_label_protocol':name_spans.NAME_PROTOCOL,'dose_mentions':doses}
    options=candidates(source)
    words=name_spans.source_tokens(text)
    starts={word['start']:i+1 for i,word in enumerate(words)}
    ends={word['end']:i+1 for i,word in enumerate(words)}
    result=dict(payload)
    for field in FIELDS:
        allowed={(starts[s['start']],ends[s['end']]) for s in options[field]} if names else set()
        result[f'{field}_spans']=[pair for pair in payload[f'{field}_spans'] if tuple(pair) in allowed]
    return result


def link_attributes(text,names,doses,routes,frequencies,field):
    entities=[(s,tag) for tag,values in [('DOSE',doses),('ROUTE',routes),('FREQUENCY',frequencies)] for s in values]
    attributes=routes if field=='route' else frequencies
    links=[]
    for index,attribute in enumerate(attributes):
        matches=[]
        for med,name in enumerate(names):
            if name['end']<=attribute['start']:a,b=name['end'],attribute['start']
            elif attribute['end']<=name['start']:a,b=attribute['end'],name['start']
            else:continue
            if b-a>120 or any(other['start']<b and a<other['end'] for i,other in enumerate(names) if i!=med):continue
            spans=sorted([(s,tag) for s,tag in entities if a<=s['start'] and s['end']<=b],key=lambda item:item[0]['start'])
            pieces=[];cursor=a
            for span,tag in spans:
                pieces.extend([text[cursor:span['start']],f'<{tag}>']);cursor=span['end']
            pieces.append(text[cursor:b]);bridge=''.join(pieces)
            pattern=_BRIDGE if name['end']<=attribute['start'] else _BEFORE_NAME
            if re.search(r'[\n\r;.!?]',bridge) or not pattern.fullmatch(bridge):continue
            if med and name['end']<=attribute['start']:
                previous=text[names[med-1]['end']:name['start']]
                if re.fullmatch(r'\s*(?:,|and|or|and/or)\s*',previous,re.I):continue
            matches.append(med)
        if len(matches)==1:links.append({'medication_index':matches[0],f'{field}_index':index})
    return links


def output(text,payload,dose_linking_rule=medication_doses.LINKING_RULE):
    result=medication_doses.output(text,payload,linking_rule=dose_linking_rule)
    names,doses=result['medication_names'],result['dose_spans']
    routes=attribute_spans(text,payload,'route');frequencies=attribute_spans(text,payload,'frequency')
    for field,plural,spans in [('route','routes',routes),('frequency','frequencies',frequencies)]:
        links=link_attributes(text,names,doses,routes,frequencies,field)
        linked={r[f'{field}_index'] for r in links}
        result[f'{field}_spans']=spans
        result[f'unlinked_{plural}']=[span for i,span in enumerate(spans) if i not in linked]
        for i,medication in enumerate(result['medications']):
            medication[plural]=[spans[r[f'{field}_index']] for r in links if r['medication_index']==i]
    result['attribute_linking']=LINKING_RULE
    return result


def synthetic_negatives():
    return [dict(row,regimen_label_protocol=PROTOCOL,route_mentions=[],route_relations=[],frequency_mentions=[],frequency_relations=[])
            for row in medication_doses.synthetic_negatives()]


def score(references,predictions,linking_rule=medication_doses.LINKING_RULE):
    result=medication_doses.score(references,predictions,linking_rule=linking_rule)
    counts={key:[0,0,0] for key in ('routes','frequencies','medication_route_pairs','medication_frequency_pairs','complete_records')}
    def key(span):return span['start'],span['end']
    for record,payload in zip(references,predictions,strict=True):
        reference_spans(record)
        expected_names=name_spans.resolve_response(record,json.dumps(name_spans.reference_payload(record)))
        actual=output(record['text'],payload,dose_linking_rule=linking_rule)
        gold_records={key(name):{'doses':[],'routes':[],'frequencies':[]} for name in expected_names}
        for field,plural in [('dose','doses'),('route','routes'),('frequency','frequencies')]:
            for relation in record[f'{field}_relations']:
                gold_records[key(expected_names[relation['medication_index']])][plural].append(key(record[f'{field}_mentions'][relation[f'{field}_index']]))
        expected={'complete_records':{(n,*(tuple(sorted(fields[k])) for k in ('doses','routes','frequencies'))) for n,fields in gold_records.items()}}
        observed={'complete_records':{(key(m['name']),*(tuple(sorted(key(s) for s in m[k])) for k in ('doses','routes','frequencies'))) for m in actual['medications']}}
        for field,plural in [('route','routes'),('frequency','frequencies')]:
            expected[plural]={key(s) for s in record[f'{field}_mentions']}
            observed[plural]={key(s) for s in actual[f'{field}_spans']}
            expected[f'medication_{field}_pairs']={(key(expected_names[r['medication_index']]),key(record[f'{field}_mentions'][r[f'{field}_index']])) for r in record[f'{field}_relations']}
            observed[f'medication_{field}_pairs']={(key(m['name']),key(s)) for m in actual['medications'] for s in m[plural]}
        for kind in counts:
            for i,value in enumerate((len(expected[kind]&observed[kind]),len(observed[kind]-expected[kind]),len(expected[kind]-observed[kind]))):counts[kind][i]+=value
    for kind,(tp,fp,fn) in counts.items():
        result[kind]={'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else None,
                      'recall':tp/(tp+fn) if tp+fn else None,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None}
    return result
