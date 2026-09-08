from pathlib import Path
import json, math
from collections import Counter
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from PIL import Image as PILImage

ROOT=Path(r'C:\Users\user\BIO MEDICAL ROBOTICS Dropbox\Elisheva Shiri Decktor\BGU\Codes\Parallel_Heptics\analysis\Kinematics')
RF=ROOT/'result_fillter'
OUTDIR=ROOT/'output'/'pdf'
OUTDIR.mkdir(parents=True, exist_ok=True)
PDF=OUTDIR/'kinematics_result_fillter_report.pdf'
MD=OUTDIR/'kinematics_result_fillter_report_summary.md'
MANIFEST=OUTDIR/'kinematics_result_fillter_plot_manifest.csv'
summary=json.loads((ROOT/'kinematics_report_summary.json').read_text(encoding='utf-8'))
funcs=json.loads((ROOT/'output_function_catalog.json').read_text(encoding='utf-8'))
rows=[]
for g in ['L_N_E','L_E','N_E']:
    for q in sorted((RF/g/'figures').rglob('*.png')):
        rel=q.relative_to(RF/g)
        rows.append({'group':g,'relative_path':str(rel),'category':rel.parts[1] if len(rel.parts)>1 else '', 'file':q.name, 'bytes':q.stat().st_size})
pd.DataFrame(rows).to_csv(MANIFEST,index=False)
styles=getSampleStyleSheet()
for st in [('TitleCenter','Title',20,24,TA_CENTER,'#000000'),('H1x','Heading1',15,18,None,'#1f4e79'),('H2x','Heading2',12.5,15,None,'#365f91'),('Bodyx','BodyText',8.8,11.2,None,'#000000'),('Smallx','BodyText',7.2,8.8,None,'#333333'),('Caption','BodyText',7.2,8.8,TA_CENTER,'#555555'),('Bulletx','BodyText',8.8,11.2,None,'#000000')]:
    name,parent,size,lead,align,color=st
    kwargs={'name':name,'parent':styles[parent],'fontSize':size,'leading':lead,'spaceAfter':4,'textColor':colors.HexColor(color)}
    if align is not None: kwargs['alignment']=align
    if name=='Bulletx': kwargs.update({'leftIndent':12,'firstLineIndent':-7,'spaceAfter':2})
    styles.add(ParagraphStyle(**kwargs))

def clean(s):
    s=str(s)
    trans={ord('\u2013'):'-',ord('\u2014'):'-',ord('\u2212'):'-',ord('\u2011'):'-',ord('\u00b1'):'+/-'}
    s=s.translate(trans)
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def p(txt,style='Bodyx'): return Paragraph(clean(txt), styles[style])
def fmt(x,nd=3):
    if x is None: return ''
    try: x=float(x)
    except Exception: return str(x)
    if math.isnan(x) or math.isinf(x): return ''
    if abs(x)>=1e6: return f'{x:.3g}'
    if abs(x)>=1000: return f'{x:,.1f}'
    if abs(x)>=100: return f'{x:.1f}'
    if abs(x)>=10: return f'{x:.2f}'
    if abs(x)>=1: return f'{x:.3f}'
    return f'{x:.4f}'
def table(data,col_widths=None,font=7,header=True,repeatRows=1):
    pdata=[]
    for row in data:
        prow=[]
        for c in row:
            prow.append(c if hasattr(c,'wrap') else Paragraph(clean(c), styles['Smallx']))
        pdata.append(prow)
    t=Table(pdata,colWidths=col_widths,repeatRows=repeatRows if header else 0,hAlign='LEFT')
    ts=[('FONTSIZE',(0,0),(-1,-1),font),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#dddddd')),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]
    if header: ts += [('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d9eaf7'))]
    t.setStyle(TableStyle(ts)); return t
def image_flowable(path,max_w,max_h):
    path=Path(path)
    if not path.exists(): return p('Missing image: '+str(path),'Smallx')
    with PILImage.open(path) as im: w,h=im.size
    scale=min(max_w/w,max_h/h)
    return Image(str(path), width=w*scale, height=h*scale)
def add_heading(story,text,level=1): story.append(p(text,'H1x' if level==1 else 'H2x'))

story=[]
story += [p('Kinematics Analysis Report: result_fillter','TitleCenter'), p('Main analysis: L_N_E. Comparison groups: L_E and N_E.','H2x'), p(f'Generated from local outputs in: {RF}','Smallx'), p('Report generated on 2026-08-15. Evidence comes from generated CSV tables, PNG files, and the local kinematics_analysis.py function catalog. No external assumptions were added.','Bodyx'), Spacer(1,0.15*inch)]
add_heading(story,'1. Executive bottom line')
subjects=summary['subjects_by_group']; trials=summary['trial_segments_by_group']
story.append(p(f'L_N_E combines two matched experiment groups: L_E (n={subjects.get("L_E")}) and N_E (n={subjects.get("N_E")}), with {trials.get("L_E"):,} trial-segment rows per group ({summary["rows_total"]:,} total). The dataset spans {len(summary["stiffness_values"])} stiffness levels ({", ".join(map(lambda x: fmt(x,0), summary["stiffness_values"]))}) and four finger conditions ({", ".join(summary["fingers"])}).'))
km={r['metric']:r for r in summary['key_metric_group_means']}
def metric_sentence(metric,label,unit=''):
    r=km[metric]; return f'{label}: L_E={fmt(r["L_E"])}{unit}, N_E={fmt(r["N_E"])}{unit}, L_E minus N_E={fmt(r["L_E_minus_N_E"])}{unit} ({fmt(r["percent_diff_vs_N_E"],1)}% vs N_E).'
for bullet in ['The largest and most consistent L_E vs N_E difference is spatial scale: L_E moves farther from center and reaches larger maximum radius than N_E.', metric_sentence('mean_r_workspace_cm','Mean radius from workspace center',' cm'), metric_sentence('mean_max_r_workspace_cm','Mean maximum radius',' cm'), metric_sentence('mean_path_length_cm','Mean path length',' cm'), metric_sentence('mean_speed_cm_s','Mean speed',' cm/s'), metric_sentence('mean_curvature_1_cm','Mean curvature',' 1/cm')+' Lower curvature in L_E indicates straighter, less curved paths relative to N_E.', metric_sentence('success_rate','Success rate','')+' N_E is slightly higher by about 3 percentage points.', 'Jerk and normalized jerk are much higher in L_E. This points to less smooth motion in the L workspace, but normalized jerk is sensitive to path/time scaling and should be interpreted together with path length and speed.']:
    story.append(p('- '+bullet,'Bulletx'))
add_heading(story,'2. Evidence inventory')
plot_counts=summary['plot_counts']; csv_counts=summary['csv_counts']
story.append(table([['Group','Subjects','Trial rows','CSV tables','PNG plots','Main role'],['L_N_E',str(summary['subjects_total']),f'{summary["rows_total"]:,}',str(csv_counts['L_N_E']['total_csv']),str(plot_counts['L_N_E']['total_png']),'Primary combined analysis'],['L_E',str(subjects.get('L_E')),f'{trials.get("L_E"):,}',str(csv_counts['L_E']['total_csv']),str(plot_counts['L_E']['total_png']),'Comparison group'],['N_E',str(subjects.get('N_E')),f'{trials.get("N_E"):,}',str(csv_counts['N_E']['total_csv']),str(plot_counts['N_E']['total_png']),'Comparison group']], [1.0*inch,0.7*inch,0.85*inch,0.8*inch,0.8*inch,2.3*inch]))
story.append(p(f'Complete plot manifest saved alongside this PDF: {MANIFEST}', 'Smallx'))
setup_path=RF/'L_N_E'/'csv'/'other'/'experiment_setup_context.csv'
if setup_path.exists():
    add_heading(story,'3. Experiment setup context found in outputs')
    setup=pd.read_csv(setup_path)
    story.append(table([list(setup.columns)]+setup.astype(str).values.tolist(), [0.8*inch,0.75*inch,0.75*inch,0.75*inch,1.1*inch,1.25*inch,2.2*inch], font=6.6))
    story.append(p('Interpretation note: L and N appear to be workspace/setup factors documented in the output table: L uses an 80 x 60 cm lab/airslide field with right-side camera, while N uses a 60 x 45 cm natural field with left-side camera. The report does not infer demographics beyond these labels.','Smallx'))
add_heading(story,'4. Main L_N_E quantitative findings')
metric_labels={'success_rate':'Success rate','mean_speed_cm_s':'Mean speed (cm/s)','mean_acceleration_cm_s2':'Mean acceleration (cm/s2)','mean_jerk_cm_s3':'Mean jerk (cm/s3)','mean_normalized_jerk_cost_cm':'Normalized jerk cost','mean_curvature_1_cm':'Mean curvature (1/cm)','mean_path_length_cm':'Path length (cm)','mean_straightness_index_cm':'Straightness index','mean_max_r_workspace_cm':'Max radius (cm)','mean_r_workspace_cm':'Mean radius (cm)','mean_thumb_active_span_cm':'Thumb-active span (cm)'}
interpret={'success_rate':'N_E modestly higher.','mean_speed_cm_s':'L_E faster.','mean_acceleration_cm_s2':'L_E slightly higher.','mean_jerk_cm_s3':'L_E much jerkier.','mean_normalized_jerk_cost_cm':'L_E much higher; scale-sensitive.','mean_curvature_1_cm':'N_E more curved.','mean_path_length_cm':'L_E longer paths.','mean_straightness_index_cm':'Very small absolute difference.','mean_max_r_workspace_cm':'L_E much larger excursion.','mean_r_workspace_cm':'L_E farther from center.','mean_thumb_active_span_cm':'Very similar.'}
data=[['Metric','L_E mean','N_E mean','L_E - N_E','% vs N_E','Interpretation']]
for r in summary['key_metric_group_means']:
    data.append([metric_labels.get(r['metric'],r['metric']),fmt(r.get('L_E')),fmt(r.get('N_E')),fmt(r.get('L_E_minus_N_E')),fmt(r.get('percent_diff_vs_N_E'),1)+'%',interpret.get(r['metric'],'')])
story.append(table(data,[1.45*inch,0.72*inch,0.72*inch,0.72*inch,0.67*inch,2.1*inch],font=6.8))
add_heading(story,'5. L_E vs N_E differences from generated between-group table')
story.append(p('The generated comparison table reports L_E - N_E as group_b minus group_a with N_E as the reference group in the all-condition rows. Positive values mean L_E > N_E; negative values mean L_E < N_E.'))
data=[['Metric','N_E mean','L_E mean','L_E - N_E','Cohen d','Direction']]
for r in summary['between_group_all_metrics'][:18]:
    diff=r['mean_difference_b_minus_a']; data.append([metric_labels.get(r['metric'],r['metric']),fmt(r['mean_a']),fmt(r['mean_b']),fmt(diff),fmt(r['cohens_d_b_minus_a']),'L_E higher' if diff>0 else 'N_E higher'])
story.append(table(data,[1.65*inch,0.75*inch,0.75*inch,0.85*inch,0.7*inch,1.15*inch],font=6.8))
story.append(p('Strongest all-condition effects are spatial: mean maximum radius, mean radius, and workspace-normalized radius are much larger for L_E. Curvature is higher for N_E. Speed/path-length are higher for L_E. Success is modestly higher for N_E.'))
add_heading(story,'6. Stiffness effects inside L_N_E')
data=[['Stiffness','Trials','Subjects','Success','Speed cm/s','Accel cm/s2','Path cm']]
for r in summary['stiffness_summary']:
    data.append([fmt(r['stiffness_value'],0),f"{int(r['n_trials']):,}",str(int(r['n_subjects'])),fmt(r.get('success_rate')),fmt(r.get('mean_speed_cm_s')),fmt(r.get('mean_acceleration_cm_s2')),fmt(r.get('mean_path_length_cm'))])
story.append(table(data,[0.75*inch,0.75*inch,0.65*inch,0.7*inch,0.85*inch,0.85*inch,0.8*inch],font=7))
story.append(p('Across stiffness levels, the generated tables allow per-stiffness interpretation without collapsing subject identity. The high-level summary does not show a single monotonic stiffness effect dominating the much larger L_E vs N_E spatial setup difference.'))
add_heading(story,'7. Velocity, acceleration, z-lift, and success-linked findings')
data=[['Stiffness','n obs','n subj','Mean velocity cm/s','Mean 3D proxy velocity cm/s','Mean accel cm/s2','Mean 3D proxy accel cm/s2']]
for r in summary['velocity_stiffness']:
    data.append([fmt(r['stiffness_value'],0),str(int(r['n_observations'])),str(int(r['n_subjects'])),fmt(r['mean_velocity_cm_s']),fmt(r['mean_velocity_3d_proxy_cm_s']),fmt(r['mean_acceleration_cm_s2']),fmt(r['mean_acceleration_3d_proxy_cm_s2'])])
story.append(table(data,[0.65*inch,0.65*inch,0.55*inch,1.0*inch,1.15*inch,1.0*inch,1.2*inch],font=6.5))
story.append(Spacer(1,0.08*inch))
data=[['Stiffness','Mean z lift cm','SEM z lift cm','Max z lift cm','Success']]
for r in summary['z_by_stiffness']:
    data.append([fmt(r['stiffness_value'],0),fmt(r['mean_side_z_lift_cm']),fmt(r['sem_side_z_lift_cm']),fmt(r['max_side_z_lift_cm']),fmt(r['success_rate'])])
story.append(table(data,[0.75*inch,1.0*inch,1.0*inch,1.0*inch,0.8*inch],font=7))
story.append(p('Z/lift outputs are derived from side-camera tracking and are corrected for side-camera orientation. They should be read as proxy vertical/lift measures unless independently calibrated.','Smallx'))
data=[['Metric','n paired','Success - failure','Cohen dz','p sign-flip']]
for r in summary['success_contrasts_top'][:10]:
    data.append([metric_labels.get(r['metric'],r['metric']),str(int(r['n_paired_observations'])),fmt(r['mean_difference']),fmt(r['cohens_dz']),fmt(r['sign_flip_p'])])
story.append(table(data,[2.1*inch,0.8*inch,1.0*inch,0.8*inch,0.8*inch],font=6.8))
story.append(p('Success-linked contrasts compare successful and unsuccessful trials within available subject/finger observations. Positive values mean successful trials had larger values for that metric.','Smallx'))
add_heading(story,'8. What each analysis function does')
story.append(p('The source file is kinematics_analysis.py. It contains a large notebook-backed pipeline. The table below lists every top-level function found by AST parsing with line ranges and first docstring sentence where available. Helper functions beginning with underscore are implementation support; non-underscore functions are direct pipeline or public helpers.'))
key_funcs=['copy_group_tree_for_subject_selection','save_selected_kinematic_tree','organize_kinematic_results_tree','discover_trials','compute_tracking_kinematics','estimate_side_video_z','summarize_kinematics','compute_experiment_group_comparisons','compute_expanded_kinematic_scope_tables','save_interaction_filtered_kinematic_outputs','compute_motor_control_comparisons','compute_success_kinematic_z_analysis','compute_trajectory_similarity_analysis','compute_subject_spatial_trajectory_analysis','compute_subject_velocity_acceleration_analysis','add_velocity_decomposition_columns','compute_3d_proxy_kinematics','compute_hand_orientation_plane_analysis','save_hand_orientation_plane_figures','save_movement_cycle_hand_angle_figures','save_kinematic_figures','save_individual_subject_reports','analysis_manifest']
fd={f['name']:f for f in funcs}
for name in key_funcs:
    if name in fd:
        f=fd[name]; story.append(p(f'- {name} (lines {f["line"]}-{f["end_line"]}): {f.get("doc") or "No docstring summary."}','Bulletx'))
story.append(PageBreak())
add_heading(story,'Appendix A. Full function catalog')
for ci in range(0,len(funcs),34):
    data=[['Function','Lines','Purpose / first docstring sentence']]
    for f in funcs[ci:ci+34]: data.append([f['name'],f"{f['line']}-{f['end_line']}",f.get('doc') or 'No docstring summary.'])
    story.append(table(data,[1.7*inch,0.65*inch,4.0*inch],font=6.4))
    if ci+34<len(funcs): story.append(PageBreak())
story.append(PageBreak())
add_heading(story,'Appendix B. Output table and plot inventory')
for g in ['L_N_E','L_E','N_E']:
    add_heading(story,f'{g} inventory',2)
    csvs=[]
    for fp in sorted((RF/g/'csv').rglob('*.csv')):
        try:
            cols=pd.read_csv(fp,nrows=0).columns.tolist(); n=sum(1 for _ in open(fp,'rb'))-1
        except Exception: cols=[]; n='?'
        csvs.append([str(fp.relative_to(RF/g)),str(n),str(len(cols)),', '.join(cols[:8])])
    story.append(table([['CSV path','Rows','Cols','First columns']]+csvs[:80],[2.3*inch,0.6*inch,0.45*inch,3.0*inch],font=5.8))
    story.append(p(f'Plot counts by top category: {plot_counts[g]["by_top_category"]}. Complete plot list is in {MANIFEST.name}.','Smallx'))
    if g!='N_E': story.append(PageBreak())
story.append(PageBreak())
add_heading(story,'Appendix C. Matched plots for all three groups')
story.append(p('Each page compares the same plot family across L_N_E, L_E, and N_E when all three images exist. This attaches all 27 aggregate comparison plots from L_E and all 27 from N_E, plus the matching L_N_E versions.'))
le_pngs=sorted((RF/'L_E'/'figures').rglob('*.png'))
for le in le_pngs:
    rel=le.relative_to(RF/'L_E'/'figures')
    rels=str(rel)
    ne=RF/'N_E'/'figures'/rels.replace('group_L_E','group_N_E').replace('experiment_group_L_E','experiment_group_N_E')
    lne=RF/'L_N_E'/'figures'/rels.replace('group_L_E','group_L_N_E').replace('experiment_group_L_E','experiment_group_L_N_E')
    add_heading(story,rels,2)
    imgs=[]; labels=[]
    for lab,path in [('L_N_E',lne),('L_E',le),('N_E',ne)]:
        imgs.append(image_flowable(path,2.25*inch,2.0*inch) if path.exists() else p('Missing','Caption')); labels.append(p(lab,'Caption'))
    story.append(table([imgs,labels],[2.3*inch,2.3*inch,2.3*inch],header=False))
    story.append(p('Source paths: '+' | '.join([f'{lab}: {path.relative_to(RF) if path.exists() else "missing"}' for lab,path in [('L_N_E',lne),('L_E',le),('N_E',ne)]]),'Smallx'))
    story.append(PageBreak())
add_heading(story,'Appendix D. Additional L_N_E primary plots')
patterns=['figures/acceleration/average_acceleration_vs_stiffness.png','figures/acceleration/L_N_E_acceleration_by_finger.png','figures/other/motor_control_finger_metric_summary.png','figures/other/motor_control_stiffness_metric_summary.png','figures/other/motor_control_within_finger_stiffness_slopes.png','figures/success/success_by_dominant_direction.png','figures/success/success_linked_kinematic_z_contrasts.png','figures/success/success_vs_distance_from_center.png','figures/trajectories/between_finger_trajectory_distance.png','figures/trajectories/max_3d_excursion_by_direction_stiffness_bin.png','figures/trajectories/mean_3d_velocity_by_direction_stiffness_bin.png','figures/trajectories/path_length_3d_by_direction_stiffness_bin.png','figures/trajectories/movement_orientation/all_xy_trajectories_with_finger_average.png','figures/trajectories/movement_orientation/all_xy_trajectories_with_stiffness_average.png','figures/trajectories/z_lift/side_z_lift_vs_stiffness.png','figures/trajectories/z_lift/side_z_lift_timecourse_by_stiffness.png','figures/velocity/others/velocity_finger_influence_summary.png','figures/velocity/others/velocity_stiffness_influence_summary.png','figures/velocity/others/velocity_time_influence_summary.png','figures/velocity/magnitude/all_velocity_magnitude_vs_time_by_finger_median.png']
existing=[]
for pat in patterns:
    path=RF/'L_N_E'/pat
    if path.exists(): existing.append(path)
for glob in ['figures/velocity/components/all_velocity_xyz_vs_time_by_finger_median*.png','figures/velocity/radial/all_radial_velocity_xyz_vs_time_by_finger_median*.png','figures/velocity/tangential/all_tangential_velocity_xyz_vs_time_by_finger_median*.png','figures/acceleration/median_by_stiffness/all_acceleration_xyz_vs_time_by_finger_median*.png']:
    for path in sorted((RF/'L_N_E').glob(glob))[:4]:
        if path not in existing: existing.append(path)
for i in range(0,len(existing),2):
    pair=existing[i:i+2]; cells=[]; caps=[]
    for path in pair:
        cells.append(image_flowable(path,3.1*inch,3.0*inch)); caps.append(p(str(path.relative_to(RF/'L_N_E')),'Caption'))
    if len(cells)==1: cells.append(''); caps.append('')
    story.append(table([cells,caps],[3.35*inch,3.35*inch],header=False)); story.append(Spacer(1,0.08*inch))
    if i % 4 == 2: story.append(PageBreak())
story.append(PageBreak())
add_heading(story,'Appendix E. Interpretation caveats and validation')
for b in ['This PDF reports generated outputs; it does not rerun the full original kinematics pipeline.','L_N_E is treated as the main combined analysis; L_E and N_E are interpreted as comparison groups because their folders contain the aggregate comparison outputs requested by the user.','L_E and N_E differ in documented workspace dimensions and side-camera placement. Camera/setup normalization columns exist, but any residual setup effects should be considered when interpreting group differences.','All plot paths are preserved in the manifest CSV next to this PDF so omitted subject-level plots remain traceable.','Subagent exploration was attempted but failed because this ChatGPT account does not support the configured spark model for the explore role; the report was generated by direct local inspection instead.']:
    story.append(p('- '+b,'Bulletx'))
md=['# Kinematics result_fillter report summary\n','PDF: '+str(PDF)+'\n','Main bottom line: L_E shows larger spatial excursion, path length, speed, and jerk; N_E shows higher curvature and slightly higher success. See PDF for tables and plot appendix.\n','Key metric rows:\n']
for r in summary['key_metric_group_means']: md.append(f'- {r["metric"]}: L_E {fmt(r["L_E"])}; N_E {fmt(r["N_E"])}; diff {fmt(r["L_E_minus_N_E"])} ({fmt(r["percent_diff_vs_N_E"],1)}%).')
MD.write_text('\n'.join(md),encoding='utf-8')
def footer(canvas,doc):
    canvas.saveState(); canvas.setFont('Helvetica',7); canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(doc.leftMargin,0.35*inch,'Kinematics result_fillter report')
    canvas.drawRightString(doc.pagesize[0]-doc.rightMargin,0.35*inch,f'Page {doc.page}')
    canvas.restoreState()
doc=SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=0.45*inch,leftMargin=0.45*inch,topMargin=0.45*inch,bottomMargin=0.55*inch)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(PDF)
print(MD)
print(MANIFEST)
