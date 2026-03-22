import datetime as dt
from pathlib import Path
import pandas as pd
from data.data_fetcher import fetch_intraday_ohlcv
from engines.sessions import compute_session_stats
from engines.patterns import detect_patterns
from engines.zones import build_htf_zones
from engines.strategy_playbook import build_strategy_playbook
REQUESTED_SYMBOL='NQH26'; FETCH_SYMBOL='NQ=F'; POINT_VALUE=20.0; HISTORY=Path('history')
saved_dates=[]
for p in HISTORY.glob(f'*_{REQUESTED_SYMBOL}.json'):
    try: saved_dates.append(dt.datetime.strptime(p.stem.split('_')[0], '%Y-%m-%d').date())
    except Exception: pass
saved_dates=sorted(set(saved_dates))[-10:]
if not saved_dates:
    print('SUMMARY'); print({'days_analyzed':0}); raise SystemExit
start_date=saved_dates[0]-dt.timedelta(days=3); end_date=saved_dates[-1]
raw_res=fetch_intraday_ohlcv(FETCH_SYMBOL,(start_date,end_date)); raw_df=raw_res[0] if isinstance(raw_res,tuple) else raw_res
if raw_df is None or raw_df.empty:
    print('SUMMARY'); print({'days_analyzed':0,'error':'no_data_downloaded'}); raise SystemExit
if 'timestamp' not in raw_df.columns:
    raw_df=raw_df.reset_index().rename(columns={raw_df.index.name or 'index':'timestamp'})
raw_df['timestamp']=pd.to_datetime(raw_df['timestamp']); raw_df=raw_df.sort_values('timestamp').reset_index(drop=True)
def day_slice(df,d): return df[df['timestamp'].dt.date==d].copy()
def eval_trade(df_day,tt,direction,entry,targets):
    tlist=[]
    for t in targets or []:
        try:px=float(t.get('Price'))
        except Exception:continue
        if direction=='Bullish' and px>entry:tlist.append((px,t.get('Target','n/a')))
        elif direction=='Bearish' and px<entry:tlist.append((px,t.get('Target','n/a')))
    if not tlist:
        ex=float(df_day['close'].iloc[-1]); return None,None,False,ex,'close_no_target'
    tlist.sort(key=(lambda x:x[0]-entry) if direction=='Bullish' else (lambda x:entry-x[0]))
    tp,tn=tlist[0]; after=df_day[df_day['timestamp']>=tt]
    if after.empty:
        ex=float(df_day['close'].iloc[-1]); return tn,tp,False,ex,'close_no_bars_after_trigger'
    hit=after[after['high']>=tp] if direction=='Bullish' else after[after['low']<=tp]
    if not hit.empty:return tn,tp,True,float(tp),'target_hit'
    ex=float(df_day['close'].iloc[-1]); return tn,tp,False,ex,'close_target_not_hit'
rows=[]
for d in saved_dates:
    df_today=day_slice(raw_df,d); prev=d-dt.timedelta(days=1)
    while prev.weekday()>=5: prev-=dt.timedelta(days=1)
    df_prev=day_slice(raw_df,prev)
    if df_today.empty:
        rows.append({'date':str(d),'trade_made':False,'reason':'no_data_for_day'}); continue
    session_source=pd.concat([df_prev,df_today],ignore_index=True).sort_values('timestamp') if not df_prev.empty else df_today.copy()
    sessions=compute_session_stats(session_source,d)
    patterns=detect_patterns(sessions,df_today,df_prev if not df_prev.empty else None)
    zones=build_htf_zones(session_source) if not session_source.empty else []
    playbook=build_strategy_playbook(df_today=df_today,df_prev=df_prev if not df_prev.empty else None,sessions=sessions,patterns=patterns,zones=zones,now_et=dt.datetime.combine(d,dt.time(17,30)),whipsaw_threshold=3.0)
    decision=playbook.get('decision',{}) or {}; trigger=playbook.get('primary_trigger') or {}; targets=playbook.get('targets',[]) or []
    trade_today=decision.get('trade_today','Wait'); direction=trigger.get('direction'); entry=trigger.get('price'); tstr=trigger.get('time')
    if not (trade_today=='Yes' and direction in ('Bullish','Bearish') and entry is not None and tstr):
        rows.append({'date':str(d),'trade_made':False,'trade_today':trade_today,'trigger':trigger.get('name')}); continue
    tt=pd.to_datetime(tstr); entry=float(entry)
    tn,tp,th,ex,reason=eval_trade(df_today,tt,direction,entry,targets)
    pts=(ex-entry) if direction=='Bullish' else (entry-ex); dol=pts*POINT_VALUE
    rows.append({'date':str(d),'trade_made':True,'trigger':trigger.get('name'),'direction':direction,'entry_price':round(entry,2),'target_name':tn,'target_price':round(float(tp),2) if tp is not None else None,'target_hit':bool(th),'exit_reason':reason,'exit_price':round(float(ex),2),'pnl_points':round(float(pts),2),'pnl_dollars':round(float(dol),2)})
result_df=pd.DataFrame(rows).sort_values('date'); trades=result_df[result_df['trade_made']==True].copy() if not result_df.empty else pd.DataFrame()
print('SUMMARY')
print({'dates_analyzed':[str(d) for d in saved_dates],'days_analyzed':int(len(result_df)),'trade_days':int(len(trades)),'no_trade_days':int(len(result_df)-len(trades)),'target_hit_days':int((trades['target_hit']==True).sum()) if len(trades) else 0,'target_hit_rate':float((trades['target_hit']==True).mean()) if len(trades) else 0.0,'net_points':float(round(trades['pnl_points'].sum(),2)) if len(trades) else 0.0,'net_dollars':float(round(trades['pnl_dollars'].sum(),2)) if len(trades) else 0.0,'avg_dollars_per_trade':float(round(trades['pnl_dollars'].mean(),2)) if len(trades) else 0.0})
print('DAILY'); print(result_df.to_string(index=False))
