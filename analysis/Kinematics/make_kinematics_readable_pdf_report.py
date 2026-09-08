from pathlib import Path
import json, math
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
from PIL import Image as PILImage

ROOT=Path(r'C:\Users\user\BIO MEDICAL ROBOTICS Dropbox\Elisheva Shiri Decktor\BGU\Codes\Parallel_Heptics\analysis\Kinematics')
RF=ROOT/'result_fillter'
OUT=ROOT/'output'/'pdf'
OUT.mkdir(parents=True, exist_ok=True)
PDF=OUT/'kinematics_result_fillter_READABLE_report.pdf'
summary=json.loads((ROOT/'kinematics_report_summary.json').read_text(encoding='utf-8'))
funcs=json.loads((ROOT/'output_function_catalog.json').read_text(encoding='utf-8'))

styles=getSampleStyleSheet()
def add_style(name,parent,size,leading,space=6,color='#000000',align=None,left=0,first=0):
    kw=dict(name=name,parent=styles[parent],fontSize=size,leading=leading,spaceAfter=space,textColor=colors.HexColor(color),leftIndent=left,firstLineIndent=first)
    if align is not None: kw['alignment']=align
    styles.add(ParagraphStyle(**kw))
add_style('TitleBig','Title',26,31,12,'#12385b',TA_CENTER)
add_style('SubTitle','Heading2',16,20,10,'#1f4e79',TA_CENTER)
add_style('H1R','Heading1',18,22,10,'#1f4e79')
add_style('H2R','Heading2',14,18,7,'#365f91')
add_style('BodyR','BodyText',11,15,6,'#111111')
add_style('SmallR','BodyText',9,12,3,'#333333')
add_style('TinyR','BodyText',7.6,9,2,'#333333')
add_style('CapR','BodyText',8.2,10,3,'#555555',TA_CENTER)
add_style('BulletR','BodyText',11,15,4,'#111111',None,16,-9)

def clean(s):
    s=str(s).translate({ord('\u2013'):'-',ord('\u2014'):'-',ord('\u2212'):'-',ord('\u2011'):'-',ord('\u00b1'):'+/-'})
    return s.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
def P(s,style='BodyR'): return Paragraph(clean(s), styles[style])
def H(story,s,l=1): story.append(P(s,'H1R' if l==1 else 'H2R'))
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
def T(data,widths=None,font=9,header=True):
    pdata=[]
    for row in data:
        pdata.append([c if hasattr(c,'wrap') else Paragraph(clean(c), styles['TinyR' if font<8.5 else 'SmallR']) for c in row])
    t=Table(pdata,colWidths=widths,repeatRows=1 if header else 0,hAlign='LEFT')
    ts=[('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),0.35,colors.HexColor('#d5dbe3')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]
    if header: ts += [('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d9eaf7')),('TEXTCOLOR',(0,0),(-1,0),colors.HexColor('#12385b'))]
    t.setStyle(TableStyle(ts)); return t
def img(path,max_w,max_h):
    path=Path(path)
    if not path.exists(): return P('Missing: '+str(path),'SmallR')
    with PILImage.open(path) as im: w,h=im.size
    scale=min(max_w/w,max_h/h)
    return Image(str(path),width=w*scale,height=h*scale)

story=[]
story += [Spacer(1,0.25*inch), P('Kinematics Analysis Report','TitleBig'), P('Readable PDF version - result_fillter','SubTitle'), Spacer(1,0.15*inch), P('Main analysis group: L_N_E', 'H2R'), P('Comparison groups: L_E and N_E', 'H2R'), Spacer(1,0.15*inch), P(f'Source folder: {RF}', 'SmallR'), P('This readable edition uses larger text and larger plot panels. The complete plot manifest remains saved beside the original full report.', 'BodyR'), PageBreak()]
subjects=summary['subjects_by_group']; trials=summary['trial_segments_by_group']
H(story,'1. Bottom line - what matters most')
story.append(P(f'L_N_E contains 40 subjects total: 20 in L_E and 20 in N_E. Each comparison group has {trials.get("L_E"):,} trial-segment rows, for {summary["rows_total"]:,} total rows across the combined analysis.'))
km={r['metric']:r for r in summary['key_metric_group_means']}
def ms(metric,label,unit=''):
    r=km[metric]
    return f'{label}: L_E {fmt(r["L_E"])}{unit}; N_E {fmt(r["N_E"])}{unit}; difference {fmt(r["L_E_minus_N_E"])}{unit} ({fmt(r["percent_diff_vs_N_E"],1)}% relative to N_E).'
for b in [
    'The main L_N_E story is not a tiny statistical nuance - it is a large kinematic workspace difference between L_E and N_E.',
    ms('mean_r_workspace_cm','Mean radius from center',' cm'),
    ms('mean_max_r_workspace_cm','Maximum radius',' cm'),
    ms('mean_path_length_cm','Path length',' cm'),
    ms('mean_speed_cm_s','Movement speed',' cm/s'),
    ms('mean_curvature_1_cm','Curvature',' 1/cm') + ' N_E is more curved; L_E is straighter.',
    ms('success_rate','Success rate','') + ' N_E is modestly better.',
    'Interpretation: L_E movements are larger, faster, longer, and jerkier. N_E movements are smaller, more curved, and slightly more successful.'
]: story.append(P('- '+b,'BulletR'))

H(story,'2. Dataset and output inventory')
plot_counts=summary['plot_counts']; csv_counts=summary['csv_counts']
story.append(T([['Group','Subjects','Trial rows','CSV tables','PNG plots','Role'],['L_N_E','40',f'{summary["rows_total"]:,}',str(csv_counts['L_N_E']['total_csv']),str(plot_counts['L_N_E']['total_png']),'Main combined analysis'],['L_E',str(subjects['L_E']),f'{trials["L_E"]:,}',str(csv_counts['L_E']['total_csv']),str(plot_counts['L_E']['total_png']),'Comparison'],['N_E',str(subjects['N_E']),f'{trials["N_E"]:,}',str(csv_counts['N_E']['total_csv']),str(plot_counts['N_E']['total_png']),'Comparison']], [1.1*inch,0.9*inch,1.1*inch,1.0*inch,1.0*inch,2.0*inch]))
setup=RF/'L_N_E'/'csv'/'other'/'experiment_setup_context.csv'
if setup.exists():
    H(story,'3. Setup context from generated output',2)
    df=pd.read_csv(setup)
    story.append(T([list(df.columns)]+df.astype(str).values.tolist(), [0.8*inch,0.8*inch,0.8*inch,0.8*inch,1.2*inch,1.4*inch,3.2*inch], font=7.4))
    story.append(P('Important caveat: L and N are different documented workspace/camera setups. The report describes output differences; it does not claim that setup is the only cause.', 'SmallR'))
story.append(PageBreak())

H(story,'4. Main numerical results - easy-read table')
metric_labels={'success_rate':'Success rate','mean_speed_cm_s':'Mean speed (cm/s)','mean_acceleration_cm_s2':'Acceleration (cm/s2)','mean_jerk_cm_s3':'Jerk (cm/s3)','mean_normalized_jerk_cost_cm':'Normalized jerk cost','mean_curvature_1_cm':'Curvature (1/cm)','mean_path_length_cm':'Path length (cm)','mean_straightness_index_cm':'Straightness index','mean_max_r_workspace_cm':'Max radius (cm)','mean_r_workspace_cm':'Mean radius (cm)','mean_thumb_active_span_cm':'Thumb-active span (cm)'}
interp={'success_rate':'N_E slightly higher','mean_speed_cm_s':'L_E faster','mean_acceleration_cm_s2':'L_E slightly higher','mean_jerk_cm_s3':'L_E much jerkier','mean_normalized_jerk_cost_cm':'L_E much higher','mean_curvature_1_cm':'N_E more curved','mean_path_length_cm':'L_E longer','mean_straightness_index_cm':'small difference','mean_max_r_workspace_cm':'L_E much larger','mean_r_workspace_cm':'L_E farther from center','mean_thumb_active_span_cm':'similar'}
data=[['Metric','L_E','N_E','Difference','Meaning']]
for r in summary['key_metric_group_means']:
    data.append([metric_labels.get(r['metric'],r['metric']),fmt(r['L_E']),fmt(r['N_E']),fmt(r['L_E_minus_N_E'])+' ('+fmt(r['percent_diff_vs_N_E'],1)+'%)',interp.get(r['metric'],'')])
story.append(T(data,[2.0*inch,1.0*inch,1.0*inch,1.4*inch,2.2*inch],font=8.8))
H(story,'5. Strongest L_E vs N_E differences',2)
data=[['Metric','N_E mean','L_E mean','L_E - N_E','Effect size d']]
for r in summary['between_group_all_metrics'][:12]:
    data.append([metric_labels.get(r['metric'],r['metric']),fmt(r['mean_a']),fmt(r['mean_b']),fmt(r['mean_difference_b_minus_a']),fmt(r['cohens_d_b_minus_a'])])
story.append(T(data,[2.1*inch,1.1*inch,1.1*inch,1.2*inch,1.0*inch],font=8.5))
story.append(PageBreak())

H(story,'6. Stiffness, velocity, z-lift, and success')
story.append(P('These tables are included because they explain whether the bottom line changes across stiffness and movement dynamics.'))
data=[['Stiffness','Trials','Success','Speed cm/s','Acceleration cm/s2','Path cm']]
for r in summary['stiffness_summary']:
    data.append([fmt(r['stiffness_value'],0),f"{int(r['n_trials']):,}",fmt(r.get('success_rate')),fmt(r.get('mean_speed_cm_s')),fmt(r.get('mean_acceleration_cm_s2')),fmt(r.get('mean_path_length_cm'))])
story.append(T(data,[0.9*inch,1.0*inch,1.0*inch,1.2*inch,1.4*inch,1.0*inch]))
H(story,'Z-lift by stiffness',2)
data=[['Stiffness','Mean z-lift cm','Max z-lift cm','Success']]
for r in summary['z_by_stiffness']:
    data.append([fmt(r['stiffness_value'],0),fmt(r['mean_side_z_lift_cm']),fmt(r['max_side_z_lift_cm']),fmt(r['success_rate'])])
story.append(T(data,[1.0*inch,1.5*inch,1.4*inch,1.0*inch]))
H(story,'Success versus failure contrasts',2)
data=[['Metric','Success - failure','Effect dz','p']]
for r in summary['success_contrasts_top'][:8]:
    data.append([metric_labels.get(r['metric'],r['metric']),fmt(r['mean_difference']),fmt(r['cohens_dz']),fmt(r['sign_flip_p'])])
story.append(T(data,[2.5*inch,1.4*inch,1.0*inch,1.0*inch]))
story.append(PageBreak())

H(story,'7. Function guide - what the analysis pipeline does')
fd={f['name']:f for f in funcs}
major=[
('Selection/output organization',['copy_group_tree_for_subject_selection','save_selected_kinematic_tree','organize_kinematic_results_tree','analysis_manifest']),
('Trial discovery and raw kinematics',['discover_trials','compute_tracking_kinematics','summarize_kinematics']),
('Group comparisons and scopes',['compute_experiment_group_comparisons','compute_expanded_kinematic_scope_tables','save_interaction_filtered_kinematic_outputs']),
('Motor control and success',['compute_motor_control_comparisons','save_motor_control_figures','compute_success_kinematic_z_analysis']),
('Trajectories, velocity, and acceleration',['compute_trajectory_similarity_analysis','compute_subject_spatial_trajectory_analysis','compute_subject_velocity_acceleration_analysis','add_velocity_decomposition_columns','save_subject_velocity_acceleration_figures']),
('3D proxy and z-lift',['estimate_side_video_z','add_side_camera_angle_normalization_columns','compute_3d_proxy_kinematics','save_3d_proxy_figures']),
('Hand and movement orientation',['compute_hand_orientation_plane_analysis','save_hand_orientation_plane_figures','save_hand_orientation_axis_matrix_figures','save_movement_cycle_hand_angle_figures']),
('Plot/report generation',['save_standard_vs_comparison_velocity_figures','save_standard_vs_comparison_position_figures','save_kinematic_figures','save_individual_subject_reports'])]
for title,names in major:
    H(story,title,2)
    for name in names:
        if name in fd:
            f=fd[name]
            story.append(P(f'- {name} (lines {f["line"]}-{f["end_line"]}): {f.get("doc") or "No docstring available."}','BulletR'))
H(story,'Complete function index',2)
for i in range(0,len(funcs),26):
    data=[['Function','Lines','Purpose']]
    for f in funcs[i:i+26]:
        data.append([f['name'],f"{f['line']}-{f['end_line']}",f.get('doc') or 'No docstring.'])
    story.append(T(data,[2.2*inch,0.8*inch,6.0*inch],font=7.2))
    if i+26 < len(funcs): story.append(PageBreak())
story.append(PageBreak())

H(story,'8. Plot appendix - large matched panels')
story.append(P('For readability, each plot family uses one landscape page. L_N_E is shown large on top. L_E and N_E are shown below for comparison.'))
le_pngs=sorted((RF/'L_E'/'figures').rglob('*.png'))
for le in le_pngs:
    rel=le.relative_to(RF/'L_E'/'figures')
    rels=str(rel)
    ne=RF/'N_E'/'figures'/rels.replace('group_L_E','group_N_E').replace('experiment_group_L_E','experiment_group_N_E')
    lne=RF/'L_N_E'/'figures'/rels.replace('group_L_E','group_L_N_E').replace('experiment_group_L_E','experiment_group_L_N_E')
    H(story,rels,2)
    story.append(T([[img(lne,9.0*inch,2.65*inch)],[P('L_N_E main analysis','CapR')]], [9.4*inch], header=False))
    story.append(Spacer(1,0.05*inch))
    story.append(T([[img(le,4.55*inch,2.05*inch), img(ne,4.55*inch,2.05*inch)],[P('L_E comparison','CapR'),P('N_E comparison','CapR')]], [4.7*inch,4.7*inch], header=False))
    story.append(P('Paths: '+str(lne.relative_to(RF))+' | '+str(le.relative_to(RF))+' | '+str(ne.relative_to(RF)), 'TinyR'))
    story.append(PageBreak())

H(story,'9. Additional primary L_N_E plots')
patterns=['figures/acceleration/average_acceleration_vs_stiffness.png','figures/acceleration/L_N_E_acceleration_by_finger.png','figures/other/motor_control_finger_metric_summary.png','figures/other/motor_control_stiffness_metric_summary.png','figures/other/motor_control_within_finger_stiffness_slopes.png','figures/success/success_by_dominant_direction.png','figures/success/success_linked_kinematic_z_contrasts.png','figures/success/success_vs_distance_from_center.png','figures/trajectories/between_finger_trajectory_distance.png','figures/trajectories/max_3d_excursion_by_direction_stiffness_bin.png','figures/trajectories/mean_3d_velocity_by_direction_stiffness_bin.png','figures/trajectories/path_length_3d_by_direction_stiffness_bin.png','figures/trajectories/movement_orientation/all_xy_trajectories_with_finger_average.png','figures/trajectories/movement_orientation/all_xy_trajectories_with_stiffness_average.png','figures/trajectories/z_lift/side_z_lift_vs_stiffness.png','figures/trajectories/z_lift/side_z_lift_timecourse_by_stiffness.png','figures/velocity/others/velocity_finger_influence_summary.png','figures/velocity/others/velocity_stiffness_influence_summary.png','figures/velocity/others/velocity_time_influence_summary.png']
existing=[RF/'L_N_E'/p for p in patterns if (RF/'L_N_E'/p).exists()]
for path in existing:
    H(story,path.name,2)
    story.append(img(path,9.1*inch,5.6*inch))
    story.append(P(str(path.relative_to(RF/'L_N_E')),'CapR'))
    story.append(PageBreak())

H(story,'10. Caveats')
for b in ['This readable PDF summarizes generated results; it does not rerun the full analysis pipeline.', 'The original dense PDF and complete manifest remain in output/pdf if every image path is needed.', 'L_E and N_E differ in documented workspace size and side-camera side. Interpret group differences with that setup context in mind.', 'Large subject-level L_N_E plot sets are indexed in the manifest rather than embedded one-by-one, so this PDF remains readable.']:
    story.append(P('- '+b,'BulletR'))

def footer(canvas,doc):
    canvas.saveState(); canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#666666'))
    canvas.drawString(doc.leftMargin,0.28*inch,'Readable kinematics report')
    canvas.drawRightString(doc.pagesize[0]-doc.rightMargin,0.28*inch,f'Page {doc.page}')
    canvas.restoreState()

doc=SimpleDocTemplate(str(PDF),pagesize=landscape(A4),rightMargin=0.35*inch,leftMargin=0.35*inch,topMargin=0.35*inch,bottomMargin=0.45*inch)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(PDF)
