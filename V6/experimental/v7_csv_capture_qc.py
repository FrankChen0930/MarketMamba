"""Bounded CSV structural diagnostics; never data/research/universe admission."""
import csv,io
from collections import Counter
from datetime import date
from decimal import Decimal,InvalidOperation

CSV_HEADERS={
 '11549':['日期','證券代號','證券名稱','成交股數','成交金額','開盤價','最高價','最低價','收盤價','漲跌價差','成交筆數'],
 '11371':['資料日期','代號','名稱','收盤','漲跌','開盤','最高','最低','成交股數','成交金額','成交筆數','最後買價','最後賣價','發行股數','次日漲停價','次日跌停價']}
MISSING=frozenset({'','--','---','----','N/A','null','None'})

def diagnose(body,dataset,*,missing_tokens=MISSING):
    out={'status':'QC_FAILED','encoding':'UNKNOWN','schema':None,'rows':0,'distinct_securities':0,
         'sessions':[],'earliest':None,'latest':None,'counts':{},'exceptions':[],
         'fatal_errors':[],'volume_unit':'FIELD_LABEL_SHARES_NO_CONVERSION',
         'security_scope':'RESOURCE_SECURITIES_MIX_NOT_ADMITTED_STOCK_UNIVERSE',
         'currency_adjustment_segment':'UNKNOWN_NOT_CERTIFIED','research_admitted':False}
    counts=Counter();keys=set();securities=set();sessions=set()
    try:
        text=body.decode('utf-8-sig');out['encoding']='UTF-8'
        reader=csv.reader(io.StringIO(text,newline=''),strict=True)
        headers=next(reader);out['schema']=headers
        if headers!=CSV_HEADERS[dataset]:raise ValueError('SCHEMA_OR_FIELD_ORDER_MISMATCH')
        pnames=(['開盤價','最高價','最低價','收盤價'] if dataset=='11549' else ['開盤','最高','最低','收盤'])
        for fields in reader:
            if not fields:continue
            out['rows']+=1;reason=None
            if len(fields)!=len(headers):
                out['fatal_errors'].append('ROW_WIDTH_MISMATCH');continue
            row=dict(zip(headers,fields));rawdate=fields[0].strip();sec=fields[1].strip()
            try:
                if len(rawdate)!=7 or not rawdate.isdigit():raise ValueError('ROC_DATE_EXPECTED')
                day=date(int(rawdate[:3])+1911,int(rawdate[3:5]),int(rawdate[5:])).isoformat()
                if not sec or (day,sec) in keys:raise ValueError('EMPTY_OR_DUPLICATE_SECURITY_KEY')
                keys.add((day,sec));sessions.add(day);securities.add(sec)
            except ValueError as exc:
                out['fatal_errors'].append(str(exc));continue
            values=[row[n].strip() for n in pnames]+[row['成交股數'].strip()]
            if any(v in missing_tokens for v in values):reason='MISSING_OHLCV_NO_FILL'
            else:
                try:
                    op,hi,lo,cl,vol=[Decimal(v.replace(',','')) for v in values]
                    if not all(n.is_finite() for n in (op,hi,lo,cl,vol)) or min(op,hi,lo,cl)<=0 or vol<0 or vol!=vol.to_integral_value() or not lo<=min(op,cl)<=max(op,cl)<=hi:
                        raise ValueError('INVALID_OHLCV')
                except (InvalidOperation,ValueError):reason='INVALID_OHLCV'
            counts[reason or 'PASS']+=1
            if reason and len(out['exceptions'])<20:out['exceptions'].append({'session':day,'security_id':sec,'reason':reason})
        if not out['rows'] or not sessions:out['fatal_errors'].append('EMPTY_VALID_SUPPORT')
        out.update(counts=dict(counts),sessions=sorted(sessions),distinct_securities=len(securities))
        if sessions:out.update(earliest=min(sessions),latest=max(sessions))
        out['fatal_errors']=sorted(set(out['fatal_errors']))
        out['status']=('QC_FAILED' if out['fatal_errors'] or counts['INVALID_OHLCV'] else
                       'QC_EXCEPTIONS' if counts['MISSING_OHLCV_NO_FILL'] else 'QC_PASS_BASIC_ONLY')
    except (UnicodeError,ValueError,csv.Error,StopIteration) as exc:
        out['fatal_errors'].append(type(exc).__name__+':'+str(exc))
    return out
