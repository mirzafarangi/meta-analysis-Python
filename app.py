import os
from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import math
import copy
import requests
from IPython.display import display, HTML
import statsmodels.api as sm
from sklearn import datasets
import scipy.stats
from decimal import Decimal
import statistics
from flask_mail import Mail, Message






DEBUG = True
app = Flask(__name__)

#mail data
app.config.update(
    MAIL_SERVER= 'smtp.gmail.com',
    MAIL_PORT= 465,
    MAIL_USE_SSL= True,
    MAIL_USERNAME= 'meta.mar00@gmail.com',
    MAIL_PASSWORD= 'QWERasdfzxcv1234'
)

mail = Mail(app)



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sitemap')
def sitemap():
    return render_template('sitemap.xml')

@app.route('/effect')
def effect():
    return render_template('effect.html')

@app.route('/result1',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            study = [row['study'] for row in request.json]
            g1_sample = [row['g1_sample'] for row in request.json]
            g1_mean = [row['g1_mean'] for row in request.json]
            g1_sd = [row['g1_sd'] for row in request.json]

            g2_sample = [row['g2_sample'] for row in request.json]
            g2_mean = [row['g2_mean'] for row in request.json]
            g2_sd = [row['g2_sd'] for row in request.json]

            try:
                moderator = [row['moderator'] for row in request.json]
            except:
                moderator=['moderator']



            table=[study, g1_sample, g1_mean, g1_sd, g2_sample, g2_mean, g2_sd, moderator]


            for n, i in enumerate(moderator):

                if type(i) == str:
                    moderator[n] = 0




            df2=pd.DataFrame(table)
            df2 = df2.transpose()

            df2.index+=1

            df = pd.DataFrame()

            list = [0,1,2,3,4,5,6,7]

            for i in list:
                df[i] =  pd.to_numeric(df2[i], errors='ignore')




            df['Spooled']=(((df[1]-1)*df[3]**2+(df[4]-1)*df[6]**2)/(df[1]+df[4]-2))**0.5


            df['d']=(df[5]-df[2])/df['Spooled']
            df['g']=df['d']*(1-(3/(4*(df[1]+df[4])-9)))

            df['n']=df[1]+df[4]
            df['n_1']=(1/df[1])+(1/df[4])

            df['SEd']=((df['n_1']+(df['d']**2/(2*df['n']))))**0.5
            df['SEg']=df['SEd']*(1-3/(4*(df['n'])-9))

            df['d_lower']=df['d']-1.96*df['SEd']
            df['d_upper']=df['d']+1.96*df['SEd']

            df['g_lower']=df['g']-1.96*df['SEg']
            df['g_upper']=df['g']+1.96*df['SEg']

            df['w_d']=1/df['SEd']**2
            df['wd']=df['w_d']*df['d']

            df['w_fixed']=1/df['SEg']**2
            df['wg_fixed']=df['w_fixed']*df['g']
            df['wg2_fixed']=df['w_fixed']*df['g']**2

            g_total_fixed=np.sum(df['wg_fixed'])/np.sum(df['w_fixed'])
            se_total_fixed=(1/np.sum(df['w_fixed']))**0.5

            lower_g_fixed=g_total_fixed-1.96*se_total_fixed
            upper_g_fixed=g_total_fixed+1.96*se_total_fixed



            qq=np.sum(df['wg2_fixed'])-((np.sum(df['wg_fixed']))**2/np.sum(df['w_fixed']))
            c=np.sum(df['w_fixed'])-((np.sum(df['w_fixed']**2))/(np.sum(df['w_fixed'])))
            degf= len(df.index)-1
            if qq<=degf or degf==0:
                t2=0
                I2_fixed=0
            else:
                t2=(qq-degf)/c
                q_fixed=qq
                I2_fixed=(q_fixed-degf)/q_fixed

            df['w_random']= 1/((df['SEg']**2)+t2)
            df['wg_random']=df['w_random']*df['g']
            df['wg2_random']=df['w_random']**df['g']**2

            g_total_random=np.sum(df['wg_random'])/np.sum(df['w_random'])
            se_total_random=(1/np.sum(df['w_random']))**0.5

            lower_g_random=g_total_random-1.96*se_total_random
            upper_g_random=g_total_random+1.96*se_total_random

            df["weight(%)-fixed model"]=(df['w_fixed']/sum(df['w_fixed']))*100
            df["weight(%)-random model %"]=(df['w_random']/sum(df['w_random']))*100

            z_score_fixed=abs(g_total_fixed/se_total_fixed)
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2

            z_score_random=abs(g_total_random/se_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            #Heterogeneity
            #random model

            I2_random=I2_fixed


#Fail N safe calculation

#method The file-drawer problem (Rosenthal, 1984)/Orwin/  Fail safe for critical effect size of .20, 0.6, 0.8 = ***
            listp = df['g'].tolist()
            n_study = len(listp)
            n_ziro = n_study
            z_per = (df['d']) / df['SEd']
            fns_rosenthal= ((sum(z_per)**2)/(1.645**2)) - n_study
            t_c = scipy.stats.t.ppf(0.95,n_study)


            wi_z_05 = (sum(df['wg_fixed'])**2)/(t_c**2)-sum(df['w_fixed'])
            fns_rosenberg = (n_study*wi_z_05)/(sum(df['w_fixed']))


            p_val = p_value_fixed

            if p_val < 0.0001:
                p_val = 0.0001





            #moderator-regression analysis

            moderator_=df[7]
            effect_size=df['g']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()


            df.drop(['Spooled','d','n','n_1','SEd','d_lower','d_upper','w_d','wd','w_fixed','wg_fixed', 'wg2_fixed', 'w_random', 'wg_random', 'wg2_random'] ,inplace=True, axis=1)


            df.columns = ['Study name', 'n1', 'Mean1' , 'SD1', 'n2', 'Mean2' , 'SD2', 'Moderator Variable', 'g', 'SEg', 'g_lower', 'g_upper', 'weight(%)-fixed model', 'weight(%)-random model' ]

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_smd.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()


            #data send

            if request.headers.getlist("X-Forwarded-For"):
                ip_ad = request.headers.getlist("X-Forwarded-For")[0]

            else:
                ip_ad = request.remote_addr

            msg = Message(subject=ip_ad,
                      sender='meta.mar00@gmail.com',
                      recipients=['meta.mar00@gmail.com'])
            msg.html=df.to_html(classes="responsive-table-2 rt cf")
            with app.open_resource("results/MetaMar_result_smd.xlsx") as fp:
                 msg.attach("results/MetaMar_result_smd.xlsx", "results/MetaMar_result_smd.xlsx", fp.read())
            mail.send(msg)


            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'ave_g_fixed': float("{0:.2f}".format(g_total_fixed)),
                'ave_SEg_fixed': float("{0:.3f}".format(se_total_fixed)),
                'lower_gg_fixed': float("{0:.3f}".format(lower_g_fixed)),
                'upper_gg_fixed': float("{0:.3f}".format(upper_g_fixed)),
                'Het_fixed': 100*float("{0:.3f}".format(I2_fixed)),
                'ave_g_random': float("{0:.2f}".format(g_total_random)),
                'ave_SEg_random': float("{0:.3f}".format(se_total_random)),
                'lower_gg_random': float("{0:.3f}".format(lower_g_random)),
                'upper_gg_random': float("{0:.3f}".format(upper_g_random)),
                'Het_random': 100*float("{0:.3f}".format(I2_random)),
                't2': float("{0:.3f}".format(t2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder,
                'qq':round(qq, 2),
                'degf': degf,
                'n_study': n_study,
                't_c': round(t_c, 3),
                'fns_rosenberg': round(fns_rosenberg, 2),
                'fns_rosenthal': round(fns_rosenthal, 2),
                'p_val': round(p_val,4)
            }

            content = render_template("result1.html", **resultData)
            return jsonify({
                'content': content,
                'g_list': df['g'].tolist(),
                'study_list': df['Study name'].tolist(),
                'g_lower_list': df['g_lower'].tolist(),
                'g_upper_list': df['g_upper'].tolist(),
                'g_weight_list': df['weight(%)-random model'].tolist(),
                'weight_fixed_list': df['weight(%)-fixed model'].tolist(),
                'weight_random_list': df['weight(%)-random model'].tolist(),
                'ave_g': float("{0:.2f}".format(g_total_random)),
                'lower_g_ave': float("{0:.2f}".format(lower_g_random)),
                'upper_g_ave': float("{0:.2f}".format(upper_g_random)),
                'ave_g_fixed': float("{0:.2f}".format(g_total_fixed)),
                'lower_g_ave_fixed': float("{0:.2f}".format(lower_g_fixed)),
                'upper_g_ave_fixed': float("{0:.2f}".format(upper_g_fixed))
            })

        except Exception as error:
            print(error)
            return render_template('error_content.html')


@app.route('/return-file-smd/')
def return_file_smd():
    return send_file('results/MetaMar_result_smd.xlsx', attachment_filename='MetaMar_result_smd.xlsx')


@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        try:
            f = request.files['file']
            xl=pd.ExcelFile(f)
            df=xl.parse('Data')
            df['Moderator'] = df['Moderator'].fillna(0)
            df['subgroup'] = df['subgroup'].fillna('NAN')
            df['subgroup'] = df['subgroup'].astype(str)



            def SE(sd_c,n_c,sd_a,n_a):
                SE=(((n_c-1)*sd_c**2+(n_a-1)*sd_a**2)/(n_c+n_a-2))**0.5
                return SE

            def d(s,m_c,m_a):
                d=(m_a-m_c)/s
                return d

            def g(d,n_c,n_a):
                g=d*(1-(3/(4*(n_c+n_a)-9)))
                return g

            def i_n(n_c,n_a):
                i=(1-(3/(4*(n_c+n_a)-9)))
                return i

            subgroup_dictionary = {}
            for i in range(len(df['subgroup'])):
                if not(df['subgroup'][i] in subgroup_dictionary.keys()):
                    subgroup_dictionary[df['subgroup'][i]] = {'Study':[df['Study'][i]],'N1':[df['N1'][i]],'Mean1':[df['Mean1'][i]],'Sd1':[df['Sd1'][i]],'N2':[df['N2'][i]],'Mean2':[df['Mean2'][i]],'Sd2':[df['Sd2'][i]], 'Moderator': [df['Moderator'][i]], 'subgroup': [df['subgroup'][i]]}
                else:
                    subgroup_dictionary[df['subgroup'][i]]['Study'].append(df['Study'][i])
                    subgroup_dictionary[df['subgroup'][i]]['N1'].append(df['N1'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Mean1'].append(df['Mean1'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Sd1'].append(df['Sd1'][i])
                    subgroup_dictionary[df['subgroup'][i]]['N2'].append(df['N2'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Mean2'].append(df['Mean2'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Sd2'].append(df['Sd2'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Moderator'].append(df['Moderator'][i])
                    subgroup_dictionary[df['subgroup'][i]]['subgroup'].append(df['subgroup'][i])






            s_per_study=SE(df['Sd1'],df['N1'],df['Sd2'],df['N2'])



            d_per_study=d(s_per_study,df['Mean1'],df['Mean2'])



            g_per_study=g(d_per_study, df['N1'], df['N2'])



            i_per_study=i_n(df['N1'], df['N2'])





            n=df['N1']+df['N2']
            n_1=(1/df['N1'])+(1/df['N2'])

            se_d_per_study=((n_1+(d_per_study**2/(2*n))))**0.5
            se_g_per_study=se_d_per_study*(1-3/(4*(n)-9))

            d_lower_per_study=d_per_study-1.96*se_d_per_study
            d_upper_per_study=d_per_study+1.96*se_d_per_study

            g_lower_per_study=g_per_study-1.96*se_g_per_study
            g_upper_per_study=g_per_study+1.96*se_g_per_study

            w_s_d=1/se_d_per_study**2
            d_s=w_s_d*d_per_study

            w_s_g_fixed=1/se_g_per_study**2
            g_s_fixed=w_s_g_fixed*g_per_study
            g2_s_fixed=w_s_g_fixed*g_per_study**2


            qq=np.sum(g2_s_fixed)-((np.sum(g_s_fixed))**2/np.sum(w_s_g_fixed))
            c=np.sum(w_s_g_fixed)-((np.sum(w_s_g_fixed**2))/(np.sum(w_s_g_fixed)))
            degf= len(df.index)-1
            if qq<=degf or degf==0:
                t2=0
                I2_fixed=0
            else:
                t2=(qq-degf)/c
                q_fixed=qq
                I2_fixed=(q_fixed-degf)/q_fixed

            w_s_g_random= 1/((se_g_per_study**2)+t2)
            g_s_random=w_s_g_random*g_per_study
            g_total_random=np.sum(g_s_random)/np.sum(w_s_g_random)
            sg_total_random=(1/np.sum(w_s_g_random))**0.5

            lower_g_random=g_total_random-1.96*sg_total_random
            upper_g_random=g_total_random+1.96*sg_total_random

            g_s_random=w_s_g_random*g_per_study

            d_total=np.sum(d_s)/np.sum(w_s_d)
            s_total=(1/np.sum(w_s_d))**0.5

            lower_d=d_total-1.96*s_total
            upper_d=d_total+1.96*s_total

            g_total_fixed=np.sum(g_s_fixed)/np.sum(w_s_g_fixed)
            sg_total_fixed=(1/np.sum(w_s_g_fixed))**0.5

            lower_g_fixed=g_total_fixed-1.96*sg_total_fixed
            upper_g_fixed=g_total_fixed+1.96*sg_total_fixed

            df["Cohen's d"]=d_per_study
            df["CorrectionFactor"]=i_per_study

            df["Hedges'g (SMD)"]=g_per_study
            df["SEg"]=se_g_per_study

            df["95%CI-Lower"]= g_lower_per_study
            df["95%CI-Upper"]= g_upper_per_study

            df["weight(%)-fixed model"]=(w_s_g_fixed/sum(w_s_g_fixed))*100
            df["weight(%)-random model %"]=(w_s_g_random/sum(w_s_g_random))*100

            z_score_fixed=abs(g_total_fixed/sg_total_fixed)
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2

            z_score_random=abs(g_total_random/sg_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

#Fail N safe calculation

#method The file-drawer problem (Rosenthal, 1984)/Orwin/  Fail safe for critical effect size of .20, 0.6, 0.8 = ***
            listp = g_lower_per_study
            n_study = len(listp)
            n_ziro = n_study
            z_per = (d_per_study) / se_d_per_study
            fns_rosenthal= ((sum(z_per)**2)/(1.645**2)) - n_study
            t_c = scipy.stats.t.ppf(0.95,n_study)


            wi_z_05 = (sum(g_s_fixed)**2)/(t_c**2)-sum(w_s_g_fixed)
            fns_rosenberg = (n_study*wi_z_05)/(sum(w_s_g_fixed))


            p_val = p_value_fixed

            if p_val < 0.0001:
                p_val = 0.0001




            #Heterogeneity
            #random model

            I2_random=I2_fixed






            #moderator-regression analysis
            moderator_= df['Moderator']
            effect_size=g_per_study
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()



            df2=pd.DataFrame(index=['Fixed Effect Model','Random Effect Model'], columns=["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'Heterogeneity %'])
            df2.xs('Fixed Effect Model')["Hedges's g"]= g_total_fixed
            df2.xs('Fixed Effect Model')["SEg"]= sg_total_fixed
            df2.xs('Fixed Effect Model')["95%CI lower"]= lower_g_fixed
            df2.xs('Fixed Effect Model')["95%CI upper"]= upper_g_fixed
            df2.xs('Fixed Effect Model')['Heterogeneity %']= 100*I2_fixed
            df2.xs('Random Effect Model')["Hedges's g"]= g_total_random
            df2.xs('Random Effect Model')["SEg"]= sg_total_random
            df2.xs('Random Effect Model')["95%CI lower"]= lower_g_random
            df2.xs('Random Effect Model')["95%CI upper"]= upper_g_random
            df2.xs('Random Effect Model')['Heterogeneity %']= 100*I2_random

            #Save the analysis

            df.index+=1
            writer = pd.ExcelWriter('results/MetaMar_result_smdxl.xlsx')
            df.to_excel(writer,'Results per study')
            df2.to_excel(writer,'Total Results')
            writer.save()

            #data send
            if request.headers.getlist("X-Forwarded-For"):
                ip_ad = request.headers.getlist("X-Forwarded-For")[0]

            else:
                ip_ad = request.remote_addr
            msg = Message(subject=ip_ad,
                      sender='meta.mar00@gmail.com',
                      recipients=['meta.mar00@gmail.com'])
            msg.html=df.to_html(classes="responsive-table-2 rt cf")
            with app.open_resource("results/MetaMar_result_smdxl.xlsx") as fp:
                 msg.attach("results/MetaMar_result_smdxl.xlsx", "results/MetaMar_result_smdxl.xlsx", fp.read())
            mail.send(msg)





            dict_sub_fixed = {}
            dict_sub_random = {}
            dict_sub_g_per_study = {}
            counter= 0
            ki_ki = []
            for index, key in enumerate(subgroup_dictionary):

                counter = counter +1

                ki_ki.append(key)
                df_sub = pd.DataFrame.from_dict(list(subgroup_dictionary.values())[index])



                s_per_study_sub=SE(df_sub['Sd1'],df_sub['N1'],df_sub['Sd2'],df_sub['N2'])


                d_per_study_sub=abs(d(s_per_study_sub,df_sub['Mean1'],df_sub['Mean2']))





                g_per_study_sub=g(d_per_study_sub, df_sub['N1'], df_sub['N2'])



                i_per_study_sub=i_n(df_sub['N1'], df_sub['N2'])






                n_sub=df_sub['N1']+df_sub['N2']
                n_1_sub=(1/df_sub['N1'])+(1/df_sub['N2'])

                se_d_per_study_sub=((n_1_sub+(d_per_study_sub**2/(2*n_sub))))**0.5
                se_g_per_study_sub=se_d_per_study_sub*(1-3/(4*(n_sub)-9))

                d_lower_per_study_sub=d_per_study_sub-1.96*se_d_per_study_sub
                d_upper_per_study_sub=d_per_study_sub+1.96*se_d_per_study_sub


                g_lower_per_study_sub=g_per_study_sub-1.96*se_g_per_study_sub
                g_upper_per_study_sub=g_per_study_sub+1.96*se_g_per_study_sub

                w_s_d_sub=1/se_d_per_study_sub**2
                d_s_sub=w_s_d_sub*d_per_study_sub

                w_s_g_fixed_sub=1/se_g_per_study_sub**2
                g_s_fixed_sub=w_s_g_fixed_sub*g_per_study_sub
                g2_s_fixed_sub=w_s_g_fixed_sub*g_per_study_sub**2


                qq_sub=np.sum(g2_s_fixed_sub)-((np.sum(g_s_fixed_sub))**2/np.sum(w_s_g_fixed_sub))
                c_sub=np.sum(w_s_g_fixed_sub)-((np.sum(w_s_g_fixed_sub**2))/(np.sum(w_s_g_fixed_sub)))
                degf_sub= len(df_sub.index)-1
                if qq_sub<=degf_sub or degf_sub==0:
                    t2_sub=0
                    I2_fixed_sub=0
                else:
                    t2_sub=(qq_sub-degf_sub)/c_sub
                    q_fixed_sub=qq_sub
                    I2_fixed_sub=(q_fixed_sub-degf_sub)/q_fixed_sub

                w_s_g_random_sub= 1/((se_g_per_study_sub**2)+t2_sub)
                g_s_random_sub=w_s_g_random_sub*g_per_study_sub
                g_total_random_sub=np.sum(g_s_random_sub)/np.sum(w_s_g_random_sub)
                sg_total_random_sub=(1/np.sum(w_s_g_random_sub))**0.5

                lower_g_random_sub=g_total_random_sub-1.96*sg_total_random_sub
                upper_g_random_sub=g_total_random_sub+1.96*sg_total_random_sub

                g_s_random_sub=w_s_g_random_sub*g_per_study_sub

                d_total_sub=np.sum(d_s_sub)/np.sum(w_s_d_sub)
                s_total_sub=(1/np.sum(w_s_d_sub))**0.5

                lower_d_sub=d_total_sub-1.96*s_total_sub
                upper_d_sub=d_total_sub+1.96*s_total_sub

                g_total_fixed_sub=np.sum(g_s_fixed_sub)/np.sum(w_s_g_fixed_sub)
                sg_total_fixed_sub=(1/np.sum(w_s_g_fixed_sub))**0.5

                lower_g_fixed_sub=g_total_fixed_sub-1.96*sg_total_fixed_sub
                upper_g_fixed_sub=g_total_fixed_sub+1.96*sg_total_fixed_sub

                df_sub["Cohen's d"]=d_per_study_sub
                df_sub["CorrectionFactor"]=i_per_study_sub

                df_sub["Hedges'g (SMD)"]=g_per_study_sub
                df_sub["SEg"]=se_g_per_study_sub

                df_sub["95%CI-Lower"]= g_lower_per_study_sub
                df_sub["95%CI-Upper"]= g_upper_per_study_sub

                df_sub["weight(%)-fixed model"]=(w_s_g_fixed_sub/sum(w_s_g_fixed_sub))*100
                df_sub["weight(%)-random model %"]=(w_s_g_random_sub/sum(w_s_g_random_sub))*100

                z_score_fixed_sub=abs(g_total_fixed_sub/sg_total_fixed_sub)
                p_value_fixed_sub = scipy.stats.norm.sf(abs(z_score_fixed_sub))*2

                z_score_random_sub=abs(g_total_random_sub/sg_total_random_sub)
                p_value_random_sub = scipy.stats.norm.sf(abs(z_score_random_sub))*2


                I2_random_sub=I2_fixed_sub

                dict_sub_random [key] = [g_total_random_sub, sg_total_random_sub, lower_g_random_sub, upper_g_random_sub, z_score_random_sub,p_value_random_sub,I2_random_sub*100]
                dict_sub_g_per_study [key] = list(g_per_study_sub)

                dict_sub_fixed [key] = [g_total_fixed_sub, sg_total_fixed_sub, lower_g_fixed_sub, upper_g_fixed_sub, z_score_fixed_sub,p_value_fixed_sub,I2_fixed_sub*100]




            if counter < 2:
                df_dict_sub = pd.DataFrame({'subgroup analysis': ['Oops! There should be at least 2 subgroups for running this analysis!']})
                list_g = []
                ko_ko = []
                fvalue=0
                pvalue = 0
                list_gg_fixed = []
                gg_low_fixed = []
                gg_upp_fixed = []
                list_gg_random = []
                gg_low_random = []
                gg_upp_random = []
                df_dict_sub_random = pd.DataFrame({'subgroup analysis': ['Oops! There should be at least 2 subgroups for running this analysis!']})
                df_dict_sub_fixed = pd.DataFrame({'subgroup analysis': ['Oops! There should be at least 2 subgroups for running this analysis!']})
            else:
                ki_ki.append("total")
                dict_sub_fixed.update({"total":[g_total_fixed, sg_total_fixed, lower_g_fixed, upper_g_fixed, z_score_fixed,p_value_fixed,I2_fixed*100]})
                df_dict_sub_fixed = pd.DataFrame.from_dict(dict_sub_fixed, orient='index')
                df_dict_sub_fixed.columns = ["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'z score','p value','Heterogeneity %']
                dict_sub_random.update({"total":[g_total_random, sg_total_random, lower_g_random, upper_g_random, z_score_random,p_value_random,I2_random*100]})
                df_dict_sub_random = pd.DataFrame.from_dict(dict_sub_random, orient='index')
                df_dict_sub_random.columns = ["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'z score','p value','Heterogeneity %']





                ko_ko = ki_ki




                list_g = dict_sub_g_per_study.values()

                list_gg_fixed = list(df_dict_sub_fixed["Hedges's g"])
                gg_low_fixed = list(df_dict_sub_fixed["95%CI lower"])
                gg_upp_fixed = list(df_dict_sub_fixed["95%CI upper"])
                g_k_fixed = {k: v for k, v in zip(ki_ki,list_gg_fixed)}

                list_gg_random = list(df_dict_sub_random["Hedges's g"])
                gg_low_random = list(df_dict_sub_random["95%CI lower"])
                gg_upp_random = list(df_dict_sub_random["95%CI upper"])
                g_k_random = {k: v for k, v in zip(ko_ko,list_gg_random)}

                fvalue, pvalue = scipy.stats.f_oneway(*list_g)


                k_sub = []
                for x in dict_sub_g_per_study.values():
                    k_sub.append(len(x))
                k_sub.append(len(list(i_per_study)))

                new_k = [x-1 for x in k_sub]

                df_dict_sub_fixed.insert(0, 'k', k_sub)
                df_dict_sub_fixed.insert(8, 'df', new_k)

                df_dict_sub_random.insert(0, 'k', k_sub)
                df_dict_sub_random.insert(8, 'df', new_k)

            study_list = list(map(lambda x: str(x), df['Study'].tolist()))



            resultData = {
                'study_list': study_list,
                'ke_ke':ko_ko,
                'fvalue': fvalue,
                'pvalue': pvalue,

                'list_gg_fixed': list_gg_fixed,
                'gg_low_fixed': gg_low_fixed,
                'gg_upp_fixed': gg_upp_fixed,


                'list_gg_random': list_gg_random,
                'gg_low_random': gg_low_random,
                'gg_upp_random': gg_upp_random,
                'n_study': n_study,
                't_c': round(t_c, 3),

                'fns_rosenberg': round(fns_rosenberg, 2),
                'fns_rosenthal': round(fns_rosenthal, 2),

                'g_list': df["Hedges'g (SMD)"].tolist(),
                'seg_list': df["SEg"].tolist(),
                'g_lower_list': df["95%CI-Lower"].tolist(),
                'g_upper_list': df["95%CI-Upper"].tolist(),
                'weight_random_list': df["weight(%)-random model %"].tolist(),
                'weight_fixed_list': df["weight(%)-fixed model"].tolist(),
                'total': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'total_random': HTML(df_dict_sub_random.to_html(classes="responsive-table-2 rt cf")),
                'total_fixed': HTML(df_dict_sub_fixed.to_html(classes="responsive-table-2 rt cf")),
                'ave_g_fixed': float("{0:.2f}".format(g_total_fixed)),
                'ave_SEg_fixed': float("{0:.2f}".format(sg_total_fixed)),
                'lower_gg_fixed': float("{0:.2f}".format(lower_g_fixed)),
                'upper_gg_fixed': float("{0:.2f}".format(upper_g_fixed)),
                'ave_g_random': float("{0:.2f}".format(g_total_random)),
                'ave_SEg_random': float("{0:.3f}".format(sg_total_random)),
                'lower_gg_random': float("{0:.3f}".format(lower_g_random)),
                'upper_gg_random': float("{0:.3f}".format(upper_g_random)),
                'Het_fixed': 100 * float("{0:.3f}".format(I2_fixed)),
                'Het_random': 100 * float("{0:.2f}".format(I2_random)),
                't2': float("{0:.2f}".format(t2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_val': float("{0:.5f}".format(p_val)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'qq': float("{0:.2f}".format(qq)),
                'degf': degf,

                'moder': moder
            }


            return render_template("result2.html", **resultData)
        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/example', methods = ['GET', 'POST'])
def example():
    if 1+1 == 2:
        try:

            xl=pd.ExcelFile('static/analysis_mean.xlsx')
            df=xl.parse('Data')


            def SE(sd_c,n_c,sd_a,n_a):
                SE=(((n_c-1)*sd_c**2+(n_a-1)*sd_a**2)/(n_c+n_a-2))**0.5
                return SE

            def d(s,m_c,m_a):
                d=(m_a-m_c)/s
                return d

            def g(d,n_c,n_a):
                g=d*(1-(3/(4*(n_c+n_a)-9)))
                return g

            def i_n(n_c,n_a):
                i=(1-(3/(4*(n_c+n_a)-9)))
                return i

            subgroup_dictionary = {}
            for i in range(len(df['subgroup'])):
                if not(df['subgroup'][i] in subgroup_dictionary.keys()):
                    subgroup_dictionary[df['subgroup'][i]] = {'Study':[df['Study'][i]],'N1':[df['N1'][i]],'Mean1':[df['Mean1'][i]],'Sd1':[df['Sd1'][i]],'N2':[df['N2'][i]],'Mean2':[df['Mean2'][i]],'Sd2':[df['Sd2'][i]], 'Moderator': [df['Moderator'][i]], 'subgroup': [df['subgroup'][i]]}
                else:
                    subgroup_dictionary[df['subgroup'][i]]['Study'].append(df['Study'][i])
                    subgroup_dictionary[df['subgroup'][i]]['N1'].append(df['N1'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Mean1'].append(df['Mean1'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Sd1'].append(df['Sd1'][i])
                    subgroup_dictionary[df['subgroup'][i]]['N2'].append(df['N2'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Mean2'].append(df['Mean2'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Sd2'].append(df['Sd2'][i])
                    subgroup_dictionary[df['subgroup'][i]]['Moderator'].append(df['Moderator'][i])
                    subgroup_dictionary[df['subgroup'][i]]['subgroup'].append(df['subgroup'][i])






            s_per_study=SE(df['Sd1'],df['N1'],df['Sd2'],df['N2'])



            d_per_study=d(s_per_study,df['Mean1'],df['Mean2'])



            g_per_study=g(d_per_study, df['N1'], df['N2'])



            i_per_study=i_n(df['N1'], df['N2'])





            n=df['N1']+df['N2']
            n_1=(1/df['N1'])+(1/df['N2'])

            se_d_per_study=((n_1+(d_per_study**2/(2*n))))**0.5
            se_g_per_study=se_d_per_study*(1-3/(4*(n)-9))

            d_lower_per_study=d_per_study-1.96*se_d_per_study
            d_upper_per_study=d_per_study+1.96*se_d_per_study

            g_lower_per_study=g_per_study-1.96*se_g_per_study
            g_upper_per_study=g_per_study+1.96*se_g_per_study

            w_s_d=1/se_d_per_study**2
            d_s=w_s_d*d_per_study

            w_s_g_fixed=1/se_g_per_study**2
            g_s_fixed=w_s_g_fixed*g_per_study
            g2_s_fixed=w_s_g_fixed*g_per_study**2


            qq=np.sum(g2_s_fixed)-((np.sum(g_s_fixed))**2/np.sum(w_s_g_fixed))
            c=np.sum(w_s_g_fixed)-((np.sum(w_s_g_fixed**2))/(np.sum(w_s_g_fixed)))
            degf= len(df.index)-1
            if qq<=degf or degf==0:
                t2=0
                I2_fixed=0
            else:
                t2=(qq-degf)/c
                q_fixed=qq
                I2_fixed=(q_fixed-degf)/q_fixed

            w_s_g_random= 1/((se_g_per_study**2)+t2)
            g_s_random=w_s_g_random*g_per_study
            g_total_random=np.sum(g_s_random)/np.sum(w_s_g_random)
            sg_total_random=(1/np.sum(w_s_g_random))**0.5

            lower_g_random=g_total_random-1.96*sg_total_random
            upper_g_random=g_total_random+1.96*sg_total_random

            g_s_random=w_s_g_random*g_per_study

            d_total=np.sum(d_s)/np.sum(w_s_d)
            s_total=(1/np.sum(w_s_d))**0.5

            lower_d=d_total-1.96*s_total
            upper_d=d_total+1.96*s_total

            g_total_fixed=np.sum(g_s_fixed)/np.sum(w_s_g_fixed)
            sg_total_fixed=(1/np.sum(w_s_g_fixed))**0.5

            lower_g_fixed=g_total_fixed-1.96*sg_total_fixed
            upper_g_fixed=g_total_fixed+1.96*sg_total_fixed

            df["Cohen's d"]=d_per_study
            df["CorrectionFactor"]=i_per_study

            df["Hedges'g (SMD)"]=g_per_study
            df["SEg"]=se_g_per_study

            df["95%CI-Lower"]= g_lower_per_study
            df["95%CI-Upper"]= g_upper_per_study

            df["weight(%)-fixed model"]=(w_s_g_fixed/sum(w_s_g_fixed))*100
            df["weight(%)-random model %"]=(w_s_g_random/sum(w_s_g_random))*100

            z_score_fixed=abs(g_total_fixed/sg_total_fixed)
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2

            z_score_random=abs(g_total_random/sg_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

#Fail N safe calculation
            listp = g_lower_per_study
            n_study = len(listp)
            n_ziro = n_study
            z_per = (d_per_study) / se_d_per_study
            fns_rosenthal= ((sum(z_per)**2)/(1.645**2)) - n_study

            t_c = scipy.stats.t.ppf(0.95,n_study)

            print(t_c)


            wi_z_05 = (sum(g_s_fixed)**2)/(t_c**2)-sum(w_s_g_fixed)
            fns_rosenberg = (n_study*wi_z_05)/(sum(w_s_g_fixed))


            p_val = p_value_fixed

            if p_val < 0.0001:
                p_val = 0.0001



            #Heterogeneity
            #random model

            I2_random=I2_fixed

            #moderator-regression analysis
            moderator_=df['Moderator']
            effect_size=g_per_study
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()



            df2=pd.DataFrame(index=['Fixed Effect Model','Random Effect Model'], columns=["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'Heterogeneity %'])
            df2.xs('Fixed Effect Model')["Hedges's g"]= g_total_fixed
            df2.xs('Fixed Effect Model')["SEg"]= sg_total_fixed
            df2.xs('Fixed Effect Model')["95%CI lower"]= lower_g_fixed
            df2.xs('Fixed Effect Model')["95%CI upper"]= upper_g_fixed
            df2.xs('Fixed Effect Model')['Heterogeneity %']= 100*I2_fixed
            df2.xs('Random Effect Model')["Hedges's g"]= g_total_random
            df2.xs('Random Effect Model')["SEg"]= sg_total_random
            df2.xs('Random Effect Model')["95%CI lower"]= lower_g_random
            df2.xs('Random Effect Model')["95%CI upper"]= upper_g_random
            df2.xs('Random Effect Model')['Heterogeneity %']= 100*I2_random

            #Save the analysis

            df.index+=1
            writer = pd.ExcelWriter('results/MetaMar_result_smdxl.xlsx')
            df.to_excel(writer,'Results per study')
            df2.to_excel(writer,'Total Results')
            writer.save()





            dict_sub_fixed = {}
            dict_sub_random = {}
            dict_sub_g_per_study = {}
            counter= 0
            ki_ki = []
            for index, key in enumerate(subgroup_dictionary):

                counter = counter +1

                ki_ki.append(key)
                df_sub = pd.DataFrame.from_dict(list(subgroup_dictionary.values())[index])



                s_per_study_sub=SE(df_sub['Sd1'],df_sub['N1'],df_sub['Sd2'],df_sub['N2'])


                d_per_study_sub=d(s_per_study_sub,df_sub['Mean1'],df_sub['Mean2'])





                g_per_study_sub=g(d_per_study_sub, df_sub['N1'], df_sub['N2'])



                i_per_study_sub=i_n(df_sub['N1'], df_sub['N2'])






                n_sub=df_sub['N1']+df_sub['N2']
                n_1_sub=(1/df_sub['N1'])+(1/df_sub['N2'])

                se_d_per_study_sub=((n_1_sub+(d_per_study_sub**2/(2*n_sub))))**0.5
                se_g_per_study_sub=se_d_per_study_sub*(1-3/(4*(n_sub)-9))

                d_lower_per_study_sub=d_per_study_sub-1.96*se_d_per_study_sub
                d_upper_per_study_sub=d_per_study_sub+1.96*se_d_per_study_sub


                g_lower_per_study_sub=g_per_study_sub-1.96*se_g_per_study_sub
                g_upper_per_study_sub=g_per_study_sub+1.96*se_g_per_study_sub

                w_s_d_sub=1/se_d_per_study_sub**2
                d_s_sub=w_s_d_sub*d_per_study_sub

                w_s_g_fixed_sub=1/se_g_per_study_sub**2
                g_s_fixed_sub=w_s_g_fixed_sub*g_per_study_sub
                g2_s_fixed_sub=w_s_g_fixed_sub*g_per_study_sub**2


                qq_sub=np.sum(g2_s_fixed_sub)-((np.sum(g_s_fixed_sub))**2/np.sum(w_s_g_fixed_sub))
                c_sub=np.sum(w_s_g_fixed_sub)-((np.sum(w_s_g_fixed_sub**2))/(np.sum(w_s_g_fixed_sub)))
                degf_sub= len(df_sub.index)-1
                if qq_sub<=degf_sub or degf_sub==0:
                    t2_sub=0
                    I2_fixed_sub=0
                else:
                    t2_sub=(qq_sub-degf_sub)/c_sub
                    q_fixed_sub=qq_sub
                    I2_fixed_sub=(q_fixed_sub-degf_sub)/q_fixed_sub

                w_s_g_random_sub= 1/((se_g_per_study_sub**2)+t2_sub)
                g_s_random_sub=w_s_g_random_sub*g_per_study_sub
                g_total_random_sub=np.sum(g_s_random_sub)/np.sum(w_s_g_random_sub)
                sg_total_random_sub=(1/np.sum(w_s_g_random_sub))**0.5

                lower_g_random_sub=g_total_random_sub-1.96*sg_total_random_sub
                upper_g_random_sub=g_total_random_sub+1.96*sg_total_random_sub

                g_s_random_sub=w_s_g_random_sub*g_per_study_sub

                d_total_sub=np.sum(d_s_sub)/np.sum(w_s_d_sub)
                s_total_sub=(1/np.sum(w_s_d_sub))**0.5

                lower_d_sub=d_total_sub-1.96*s_total_sub
                upper_d_sub=d_total_sub+1.96*s_total_sub

                g_total_fixed_sub=np.sum(g_s_fixed_sub)/np.sum(w_s_g_fixed_sub)
                sg_total_fixed_sub=(1/np.sum(w_s_g_fixed_sub))**0.5

                lower_g_fixed_sub=g_total_fixed_sub-1.96*sg_total_fixed_sub
                upper_g_fixed_sub=g_total_fixed_sub+1.96*sg_total_fixed_sub

                df_sub["Cohen's d"]=d_per_study_sub
                df_sub["CorrectionFactor"]=i_per_study_sub

                df_sub["Hedges'g (SMD)"]=g_per_study_sub
                df_sub["SEg"]=se_g_per_study_sub

                df_sub["95%CI-Lower"]= g_lower_per_study_sub
                df_sub["95%CI-Upper"]= g_upper_per_study_sub

                df_sub["weight(%)-fixed model"]=(w_s_g_fixed_sub/sum(w_s_g_fixed_sub))*100
                df_sub["weight(%)-random model %"]=(w_s_g_random_sub/sum(w_s_g_random_sub))*100

                z_score_fixed_sub=abs(g_total_fixed_sub/sg_total_fixed_sub)
                p_value_fixed_sub = scipy.stats.norm.sf(abs(z_score_fixed_sub))*2

                z_score_random_sub=abs(g_total_random_sub/sg_total_random_sub)
                p_value_random_sub = scipy.stats.norm.sf(abs(z_score_random_sub))*2


                I2_random_sub=I2_fixed_sub

                dict_sub_random [key] = [g_total_random_sub, sg_total_random_sub, lower_g_random_sub, upper_g_random_sub, z_score_random_sub,p_value_random_sub,I2_random_sub*100]
                dict_sub_g_per_study [key] = list(g_per_study_sub)

                dict_sub_fixed [key] = [g_total_fixed_sub, sg_total_fixed_sub, lower_g_fixed_sub, upper_g_fixed_sub, z_score_fixed_sub,p_value_fixed_sub,I2_fixed_sub*100]




            if counter < 2:
                df_dict_sub = pd.DataFrame({'subgroup analysis': ['Oops! There should be at least 2 subgroups for running this analysis!']})
                list_g = []
                ki_ki = []
            else:
                ki_ki.append("total")
                dict_sub_fixed.update({"total":[g_total_fixed, sg_total_fixed, lower_g_fixed, upper_g_fixed, z_score_fixed,p_value_fixed,I2_fixed*100]})
                df_dict_sub_fixed = pd.DataFrame.from_dict(dict_sub_fixed, orient='index')
                df_dict_sub_fixed.columns = ["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'z score','p value','Heterogeneity %']
                dict_sub_random.update({"total":[g_total_random, sg_total_random, lower_g_random, upper_g_random, z_score_random,p_value_random,I2_random*100]})
                df_dict_sub_random = pd.DataFrame.from_dict(dict_sub_random, orient='index')
                df_dict_sub_random.columns = ["Hedges's g",'SEg', "95%CI lower", "95%CI upper", 'z score','p value','Heterogeneity %']





                for n, i in enumerate(ki_ki):

                    if type(i) == float:
                        ki_ki[n] = 'NaN'

                ko_ko = ki_ki




                list_g = dict_sub_g_per_study.values()

                list_gg_fixed = list(df_dict_sub_fixed["Hedges's g"])
                gg_low_fixed = list(df_dict_sub_fixed["95%CI lower"])
                gg_upp_fixed = list(df_dict_sub_fixed["95%CI upper"])
                g_k_fixed = {k: v for k, v in zip(ki_ki,list_gg_fixed)}

                list_gg_random = list(df_dict_sub_random["Hedges's g"])
                gg_low_random = list(df_dict_sub_random["95%CI lower"])
                gg_upp_random = list(df_dict_sub_random["95%CI upper"])
                g_k_random = {k: v for k, v in zip(ko_ko,list_gg_random)}

                fvalue, pvalue = scipy.stats.f_oneway(*list_g)


                k_sub = []
                for x in dict_sub_g_per_study.values():
                    k_sub.append(len(x))
                k_sub.append(len(list(i_per_study)))

                new_k = [x-1 for x in k_sub]

                df_dict_sub_fixed.insert(0, 'k', k_sub)
                df_dict_sub_fixed.insert(8, 'df', new_k)

                df_dict_sub_random.insert(0, 'k', k_sub)
                df_dict_sub_random.insert(8, 'df', new_k)

            study_list = list(map(lambda x: str(x), df['Study'].tolist()))
            resultData = {

                'study_list': study_list,
                'ke_ke':ko_ko,
                'fvalue': fvalue,
                'pvalue': pvalue,

                'list_gg_fixed': list_gg_fixed,
                'gg_low_fixed': gg_low_fixed,
                'gg_upp_fixed': gg_upp_fixed,


                'list_gg_random': list_gg_random,
                'gg_low_random': gg_low_random,
                'gg_upp_random': gg_upp_random,

                'fns_rosenberg': round(fns_rosenberg, 2),
                'fns_rosenthal': round(fns_rosenthal, 2),
                'n_study': n_study,
                't_c': round(t_c, 3),

                'g_list': df["Hedges'g (SMD)"].tolist(),
                'seg_list': df["SEg"].tolist(),
                'g_lower_list': df["95%CI-Lower"].tolist(),
                'g_upper_list': df["95%CI-Upper"].tolist(),
                'weight_random_list': df["weight(%)-random model %"].tolist(),
                'weight_fixed_list': df["weight(%)-fixed model"].tolist(),
                'total': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'total_random': HTML(df_dict_sub_random.to_html(classes="responsive-table-2 rt cf")),
                'total_fixed': HTML(df_dict_sub_fixed.to_html(classes="responsive-table-2 rt cf")),
                'ave_g_fixed': float("{0:.2f}".format(g_total_fixed)),
                'ave_SEg_fixed': float("{0:.2f}".format(sg_total_fixed)),
                'lower_gg_fixed': float("{0:.2f}".format(lower_g_fixed)),
                'upper_gg_fixed': float("{0:.2f}".format(upper_g_fixed)),
                'ave_g_random': float("{0:.2f}".format(g_total_random)),
                'ave_SEg_random': float("{0:.3f}".format(sg_total_random)),
                'lower_gg_random': float("{0:.3f}".format(lower_g_random)),
                'upper_gg_random': float("{0:.3f}".format(upper_g_random)),
                'Het_fixed': 100 * float("{0:.3f}".format(I2_fixed)),
                'Het_random': 100 * float("{0:.2f}".format(I2_random)),
                't2': float("{0:.2f}".format(t2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_val': float("{0:.5f}".format(p_val)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'qq': float("{0:.2f}".format(qq)),
                'degf': degf,

                'moder': moder
            }
            return render_template("result2.html", **resultData)
        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-smdxl/')
def return_file_smdxl():
    return send_file('results/MetaMar_result_smdxl.xlsx', attachment_filename='MetaMar_result_smdxl.xlsx')

@app.route('/result_corr',methods = ['POST', 'GET'])
def result_corr():
    if request.method == 'POST':
        try:
            study = [row['study'] for row in request.json]
            correlation = [row['correlation'] for row in request.json]
            sample = [row['sample'] for row in request.json]
            moderator = [row['moderator'] for row in request.json]


            table=[study, correlation, sample, moderator]
            for n, i in enumerate(moderator):

                if type(i) == str:
                    moderator[n] = 0

            df2=pd.DataFrame(table)
            df2 = df2.transpose()


            df = pd.DataFrame()
            list = [0,1,2,3]

            for i in list:
                df[i] =  pd.to_numeric(df2[i], errors='ignore')


            #fixed

            df['Var']=1/(df[2]-3)
            df['SE']=df['Var']**0.5
            df['w_fixed']=1/df['Var']




            df['r_lower']=df[1]-1.96*df['SE']
            df['r_upper']=df[1]+1.96*df['SE']

            df['Fisher z']=0.5*np.log((1+df[1])/(1-df[1]))
            df['zw_fixed']=df['Fisher z']*df['w_fixed']
            df['z2w_fixed']=df['Fisher z']**2*df['w_fixed']
            df['w2_fixed']=df['w_fixed']**2


            z_total_fixed=np.sum(df['zw_fixed'])/np.sum(df['w_fixed'])
            s_total_fixed=(1/np.sum(df['w_fixed']))**0.5
            z_low_fixed = z_total_fixed-1.96*s_total_fixed
            z_upp_fixed = z_total_fixed+1.96*s_total_fixed
            z_score_fixed = abs(z_total_fixed/s_total_fixed)
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2

            r_total_fixed= (-1+np.exp(2*z_total_fixed))/(1+np.exp(2*z_total_fixed))
            r_low_fixed = (-1+np.exp(2*z_low_fixed))/(1+np.exp(2*z_low_fixed))
            r_upp_fixed = (-1+np.exp(2*z_upp_fixed))/(1+np.exp(2*z_upp_fixed))


            #random

            sum_z2w = np.sum(df['z2w_fixed'])
            sum_zw_2 = np.sum(df['zw_fixed'])**2
            sum_w = np.sum(df['w_fixed'])

            q=sum_z2w - (sum_zw_2/sum_w)
            degf= len(df.index)-1
            c=np.sum(df['w_fixed'])-((np.sum(df['w_fixed']**2))/(np.sum(df['w_fixed'])))
            if q<=degf or degf==0:
                t2=0
                I2=0
                q=0
            else:
                t2=(q-degf)/c
                q_fixed=q
                I2=(q_fixed-degf)/q_fixed
            print(t2)
            df['w_random']=1/(df['Var']+t2)
            df['zw_random']=df['Fisher z']*df['w_random']
            df['z2w_random']=df['Fisher z']**2*df['w_random']
            df['w2_random']=df['w_random']**2


            z_total_random=np.sum(df['zw_random'])/np.sum(df['w_random'])
            print(df['w_random'])
            s_total_random=(1/np.sum(df['w_random']))**0.5
            z_low_random = z_total_random-1.96*s_total_random
            z_upp_random = z_total_random+1.96*s_total_random
            z_score_random = abs(z_total_random/s_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            r_total_random= (-1+np.exp(2*z_total_random))/(1+np.exp(2*z_total_random))
            r_low_random = (-1+np.exp(2*z_low_random))/(1+np.exp(2*z_low_random))
            r_upp_random = (-1+np.exp(2*z_upp_random))/(1+np.exp(2*z_upp_random))



            df['Weight(%)_fixed']=100*df['w_fixed']/np.sum(df['w_fixed'])
            df['Weight(%)_random']=100*df['w_random']/np.sum(df['w_random'])



            moderator_=df[3]
            effect_size=df[1]
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()



            df.drop(['Var','w_fixed','zw_fixed','z2w_fixed','w2_fixed', 'w_random', 'zw_random', 'z2w_random', 'w2_random'] ,inplace=True, axis=1)


            df.columns = ['study name', 'r', 'n', 'moderator', 'SE', 'r_lower', 'r_upper', 'Fisher z', 'Weight(%)_fixed', 'Weight(%)_random']


            df.index+=1

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_corr.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            #data send
            if request.headers.getlist("X-Forwarded-For"):
                ip_ad = request.headers.getlist("X-Forwarded-For")[0]

            else:
                ip_ad = request.remote_addr

            msg = Message(subject=ip_ad,
                      sender='meta.mar00@gmail.com',
                      recipients=['meta.mar00@gmail.com'])
            msg.html=df.to_html(classes="responsive-table-2 rt cf")
            with app.open_resource("results/MetaMar_result_corr.xlsx") as fp:
                 msg.attach("results/MetaMar_result_corr.xlsx", "results/MetaMar_result_corr.xlsx", fp.read())
            mail.send(msg)




            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),

                'ave_z_random': float("{0:.2f}".format(z_total_random)),
                'ave_r_random': float("{0:.2f}".format(r_total_random)),
                'ave_SE_random': float("{0:.3f}".format(s_total_random)),
                'lower_r_random': float("{0:.3f}".format(r_low_random)),
                'upper_r_random': float("{0:.3f}".format(r_upp_random)),
                'ave_z_fixed': float("{0:.2f}".format(z_total_fixed)),
                'ave_r_fixed': float("{0:.2f}".format(r_total_fixed)),
                'ave_SE_fixed': float("{0:.3f}".format(s_total_fixed)),
                'lower_r_fixed': float("{0:.3f}".format(r_low_fixed)),
                'upper_r_fixed': float("{0:.3f}".format(r_upp_fixed)),
                'Het': 100*float("{0:.3f}".format(I2)),
                't2': float("{0:.3f}".format(t2)),
                'p_value_random': float("{0:.4f}".format(p_value_random)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'p_value_fixed': float("{0:.4f}".format(p_value_fixed)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'moder': moder,
                'qq': float("{0:.3f}".format(q)),
                'degf': degf
            }

            content = render_template("result_corr.html", **resultData)
            return jsonify({
                'content': content,
                'r_list': df['r'].tolist(),
                'study_list': df['study name'].tolist(),
                'r_lower_list': df['r_lower'].tolist(),
                'r_upper_list': df['r_upper'].tolist(),
                'weight_random_list': df["Weight(%)_random"].tolist(),
                'weight_fixed_list': df["Weight(%)_fixed"].tolist(),
                'ave_r_random': float("{0:.2f}".format(r_total_random)),
                'lower_r_random': float("{0:.2f}".format(r_low_random)),
                'upper_r_random': float("{0:.2f}".format(r_upp_random)),
                'ave_r_fixed': float("{0:.2f}".format(r_total_fixed)),
                'lower_r_fixed': float("{0:.2f}".format(r_low_fixed)),
                'upper_r_fixed': float("{0:.2f}".format(r_upp_fixed))
            })

        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-corr/')
def return_file_corr():
    return send_file('results/MetaMar_result_corr.xlsx', attachment_filename='MetaMar_result_corr.xlsx')

@app.route('/uploader_corr', methods = ['GET', 'POST'])
def upload_file_corr():
    if request.method == 'POST':
        try:
            f = request.files['file']
            xl=pd.ExcelFile(f)
            df=xl.parse('Data')

            df.columns = ['Study','N', 'r', 'Moderator']

            #fixed

            df['Var']=1/(df['N']-3)
            df['SE']=df['Var']**0.5
            df['weight']=1/df['Var']

            df['r_lower']=df['r']-1.96*df['SE']
            df['r_upper']=df['r']+1.96*df['SE']

            df["Fisher z"]=0.5*(np.log((1+df['r'])/(1-df['r'])))

            df['z*w']=df['Fisher z']*df['weight']
            df['z2*w']=df['Fisher z']**2*df['weight']
            df['w2']=df['weight']**2

            z_total=np.sum(df['z*w'])/np.sum(df['weight'])
            s_total=(1/np.sum(df['weight']))**0.5
            z_low_fixed = z_total - 1.96*s_total
            z_upp_fixed = z_total + 1.96*s_total


            r_total= (-1+np.exp(2*z_total))/(1+np.exp(2*z_total))
            lower_r = (-1+np.exp(2*z_low_fixed))/(1+np.exp(2*z_low_fixed))
            upper_r = (-1+np.exp(2*z_upp_fixed))/(1+np.exp(2*z_upp_fixed))

            z_score = abs(z_total/s_total)
            p_value = scipy.stats.norm.sf(abs(z_score))*2

            #random

            q=np.sum(df['z2*w'])-((np.sum(df['z*w'])**2)/np.sum(df['weight']))
            c=np.sum(df['weight'])-((np.sum(df['w2']))/(np.sum(df['weight'])))
            degf= len(df.index)-1
            if q<=degf or degf==0:
                t2=0
                I2=0
                q=0
            else:
                t2=(q-degf)/c
                q_fixed=q
                I2=(q_fixed-degf)/q_fixed

            df['w_random']=1/(df['Var']+t2)
            df['z*w_random']=df['Fisher z']*df['w_random']
            df['z2*w_random']=df['Fisher z']**2*df['w_random']

            z_total_random=np.sum(df['z*w_random'])/np.sum(df['w_random'])
            s_total_random=(1/np.sum(df['w_random']))**0.5
            z_low_random = z_total_random - 1.96*s_total_random
            z_upp_random = z_total_random + 1.96*s_total_random


            r_total_random= (-1+np.exp(2*z_total_random))/(1+np.exp(2*z_total_random))
            lower_r_random = (-1+np.exp(2*z_low_random))/(1+np.exp(2*z_low_random))
            upper_r_random = (-1+np.exp(2*z_upp_random))/(1+np.exp(2*z_upp_random))

            z_score_random = abs(z_total_random/s_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2


            df['Weight(%)_fixed']=100*df['weight']/np.sum(df['weight'])
            df['Weight(%)_random']=100*df['w_random']/np.sum(df['w_random'])

            #moderator-regression analysis
            moderator_=df['Moderator']
            effect_size=df['r']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()

            df.drop(['Var','weight','z*w','z2*w','w2', 'w_random', 'z*w_random','z2*w_random'] ,inplace=True, axis=1)

            df.columns = ['study name', 'N', 'r', 'moderator', 'SE', 'r_lower', 'r_upper', 'Fisher z', 'Weight(%)_fixed','Weight(%)_random']

            df2=pd.DataFrame(index=['Fixed Effect Model','Random Effect Model'], columns=["Fisher z",'SE', 'r', "95%CI lower", "95%CI upper", 'Heterogeneity %'])
            df2.xs('Fixed Effect Model')["Fisher z"]= z_total
            df2.xs('Fixed Effect Model')["SE"]= s_total
            df2.xs('Fixed Effect Model')["r"]= r_total
            df2.xs('Fixed Effect Model')["95%CI lower"]= lower_r
            df2.xs('Fixed Effect Model')["95%CI upper"]= upper_r
            df2.xs('Fixed Effect Model')['Heterogeneity %']= 100*I2
            df2.xs('Random Effect Model')["Fisher z"]= z_total_random
            df2.xs('Random Effect Model')["SE"]= s_total_random
            df2.xs('Random Effect Model')["r"]= r_total_random
            df2.xs('Random Effect Model')["95%CI lower"]= lower_r_random
            df2.xs('Random Effect Model')["95%CI upper"]= upper_r_random
            df2.xs('Random Effect Model')['Heterogeneity %']= 100*I2

            #Save the analysis

            df.index+=1

            writer = pd.ExcelWriter('results/MetaMar_result_corrxl.xlsx')
            df.to_excel(writer,'per study')
            df2.to_excel(writer,'Total Results')

            writer.save()

            #data send
            if request.headers.getlist("X-Forwarded-For"):
                ip_ad = request.headers.getlist("X-Forwarded-For")[0]

            else:
                ip_ad = request.remote_addr
            msg = Message(subject=ip_ad,
                      sender='meta.mar00@gmail.com',
                      recipients=['meta.mar00@gmail.com'])
            msg.html=df.to_html(classes="responsive-table-2 rt cf")
            with app.open_resource("results/MetaMar_result_corrxl.xlsx") as fp:
                 msg.attach("results/MetaMar_result_corrxl.xlsx", "results/MetaMar_result_corrxl.xlsx", fp.read())
            mail.send(msg)

            study_list = list(map(lambda x: str(x), df['study name'].tolist()))
            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'study_list': study_list,
                'r_list': df['r'].tolist(),
                'r_lower_list': df['r_lower'].tolist(),
                'r_upper_list': df['r_upper'].tolist(),
                'weight_random_list': df['Weight(%)_random'].tolist(),
                'weight_fixed_list': df['Weight(%)_fixed'].tolist(),
                'ave_z': float("{0:.2f}".format(z_total)),
                'ave_r': float("{0:.2f}".format(r_total)),
                'ave_SE': float("{0:.2f}".format(s_total)),
                'lower_r': float("{0:.2f}".format(lower_r)),
                'upper_r': float("{0:.2f}".format(upper_r)),
                'Het': 100 * float("{0:.3f}".format(I2)),
                'p_value': float("{0:.6f}".format(p_value)),
                'z_score': float("{0:.3f}".format(z_score)),
                'ave_z_random': float("{0:.2f}".format(z_total_random)),
                'ave_r_random': float("{0:.2f}".format(r_total_random)),
                'ave_SE_random': float("{0:.2f}".format(s_total_random)),
                'lower_r_random': float("{0:.2f}".format(lower_r_random)),
                'upper_r_random': float("{0:.2f}".format(upper_r_random)),
                'Het': 100 * float("{0:.3f}".format(I2)),
                't2': float("{0:.3f}".format(t2)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder,
                'qq': float("{0:.3f}".format(q)),
                'degf': degf
            }

            return render_template("result_corrxls.html", **resultData)
        except Exception as error:
            print(error)
            return render_template('error_content.html')

@app.route('/return-file-corrxl/')
def return_file_corrxl():
    return send_file('results/MetaMar_result_corrxl.xlsx', attachment_filename='MetaMar_result_corrxl.xlsx')

@app.route('/result_ratios',methods = ['POST', 'GET'])
def result_ratios():
    if request.method == 'POST':
        try:
            study = [row['study'] for row in request.json]
            g1_e = [row['g1_e'] for row in request.json]
            g1_ne = [row['g1_ne'] for row in request.json]
            g2_e = [row['g2_e'] for row in request.json]
            g2_ne = [row['g2_ne'] for row in request.json]
            moderator = [row['moderator'] for row in request.json]

            table=[study, g1_e, g1_ne, g2_e, g2_ne, moderator]
            for n, i in enumerate(moderator):

                if type(i) == str:
                    moderator[n] = 0

            df2=pd.DataFrame(table)
            df2 = df2.transpose()

            df2.index+=1

            df = pd.DataFrame()

            list = [0,1,2,3,4,5]

            for i in list:
                df[i] =  pd.to_numeric(df2[i], errors='ignore')





            df['RiskRatio']=(df[1]/(df[1]+df[2]))/(df[3]/(df[3]+df[4]))
            df['LnRR']=np.log(df['RiskRatio'])
            df['V']=(1/df[1])-(1/(df[1]+df[2]))+(1/df[3])-(1/(df[3]+df[4]))
            df['SE']=df['V']**0.5
            df['lower_lnRR']=df['LnRR']-1.96*df['SE']
            df['upper_lnRR']=df['LnRR']+1.96*df['SE']
            df['lower_RR']=np.exp(df['lower_lnRR'])
            df['upper_RR']=np.exp(df['upper_lnRR'])
            df['weight_fixed model']=1/df['V']
            df['LnRR*w_fixed']=df['weight_fixed model']*df['LnRR']
            df['LnRR2*w_fixed']=df['weight_fixed model']*df['LnRR']**2


            qq=np.sum(df['LnRR2*w_fixed'])-((np.sum(df['LnRR*w_fixed']))**2/np.sum(df['weight_fixed model']))
            c=np.sum(df['weight_fixed model'])-((np.sum(df['weight_fixed model']**2))/(np.sum(df['weight_fixed model'])))
            degf= len(df.index)-1
            if qq <= degf or degf==0:
                I2=0
                t2=0
            else:
                t2=(qq-degf)/c
                I2=(qq-degf)/qq

            df["weight_random model"]= 1/(df['V']+t2)
            df['LnRR*w_random'] = df['weight_random model']*df['LnRR']
            df['weight(%)_random model']=100*df['weight_random model']/np.sum(df['weight_random model'])
            df['weight(%)_fixed model']=100*df['weight_fixed model']/np.sum(df['weight_fixed model'])

            df["Cohen'd"]=df['LnRR']*(3**0.5)*3.1415
            df['d_w'] = df["Cohen'd"] * df['weight_fixed model']


            LnRR_total_random=np.sum(df['LnRR*w_random'])/np.sum(df['weight_random model'])
            se_total_random=(1/np.sum(df['weight_random model']))**0.5
            lower_LnRR_random=LnRR_total_random-1.96*se_total_random
            upper_LnRR_random=LnRR_total_random+1.96*se_total_random
            LnRR_total_fixed=np.sum(df['LnRR*w_fixed'])/np.sum(df['weight_fixed model'])
            se_total_fixed=(1/np.sum(df['weight_fixed model']))**0.5
            lower_LnRR_fixed=LnRR_total_fixed-1.96*se_total_fixed
            upper_LnRR_fixed=LnRR_total_fixed+1.96*se_total_fixed
            RRave_random=np.exp(LnRR_total_random)
            lower_RRave_random=np.exp(lower_LnRR_random)
            upper_RRave_random=np.exp(upper_LnRR_random)
            RRave_fixed=np.exp(LnRR_total_fixed)
            lower_RRave_fixed=np.exp(lower_LnRR_fixed)
            upper_RRave_fixed=np.exp(upper_LnRR_fixed)
            z_score_fixed=abs(LnRR_total_fixed/se_total_fixed)
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2
            z_score_random=abs(LnRR_total_random/se_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

            #moderator-regression analysis
            moderator_=df[5]
            effect_size=df['LnRR']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()

            df['OddsRatio']=(df[1]*df[4])/(df[3]*df[2])
            df['LnOR']=np.log(df['OddsRatio'])
            df['V_OR']=(1/df[1])+(1/(df[2]))+(1/df[3])+(1/(df[4]))
            df['SE_OR']=df['V_OR']**0.5
            df['lower_lnOR']=df['LnOR']-1.96*df['SE_OR']
            df['upper_lnOR']=df['LnOR']+1.96*df['SE_OR']
            df['lower_OR']=np.exp(df['lower_lnOR'])
            df['upper_OR']=np.exp(df['upper_lnOR'])
            df['weight_fixed model_OR']=1/df['V_OR']
            df['d_OR']=df['LnOR']*(3**0.5)*3.1415
            df['d_w_OR'] = df['d_OR'] * df['weight_fixed model_OR']
            df['LnOR*w_fixed_OR']=df['weight_fixed model_OR']*df['LnOR']
            df['LnOR2*w_fixed_OR']=df['weight_fixed model_OR']*df['LnOR']**2


            qq_OR=np.sum(df['LnOR2*w_fixed_OR'])-((np.sum(df['LnOR*w_fixed_OR']))**2/np.sum(df['weight_fixed model_OR']))

            c_OR=np.sum(df['weight_fixed model_OR'])-((np.sum(df['weight_fixed model_OR']**2))/(np.sum(df['weight_fixed model_OR'])))
            degf_OR= len(df.index)-1

            if qq_OR <= degf_OR or degf_OR==0:
                I2_OR=0
                t2_OR=0
            else:
                t2_OR=(qq_OR-degf_OR)/c_OR
                I2_OR=(qq_OR-degf_OR)/qq_OR



            df["weight_random model_OR"]= 1/(df['V_OR']+t2_OR)
            df['LnOR*w_random']=df['weight_random model_OR']*df['LnOR']
            df['weight(%)_random model_OR']=100*df['weight_random model_OR']/np.sum(df['weight_random model_OR'])
            df['weight(%)_fixed model_OR']=100*df['weight_fixed model_OR']/np.sum(df['weight_fixed model_OR'])


            #OddsRatio

            LnOR_total_random=np.sum(df['LnOR*w_random'])/np.sum(df['weight_random model_OR'])
            se_total_random_OR=(1/np.sum(df['weight_random model_OR']))**0.5
            lower_LnOR_random=LnOR_total_random-1.96*se_total_random_OR
            upper_LnOR_random=LnOR_total_random+1.96*se_total_random_OR
            LnOR_total_fixed=np.sum(df['LnOR*w_fixed_OR'])/np.sum(df['weight_fixed model_OR'])
            se_total_fixed_OR=(1/np.sum(df['weight_fixed model_OR']))**0.5
            lower_LnOR_fixed=LnOR_total_fixed-1.96*se_total_fixed_OR
            upper_LnOR_fixed=LnOR_total_fixed+1.96*se_total_fixed_OR
            ORave_random=np.exp(LnOR_total_random)
            lower_ORave_random=np.exp(lower_LnOR_random)
            upper_ORave_random=np.exp(upper_LnOR_random)
            ORave_fixed=np.exp(LnOR_total_fixed)
            lower_ORave_fixed=np.exp(lower_LnOR_fixed)
            upper_ORave_fixed=np.exp(upper_LnOR_fixed)
            z_score_fixed_OR=abs(LnOR_total_fixed/se_total_fixed_OR)
            p_value_fixed_OR = scipy.stats.norm.sf(abs(z_score_fixed_OR))*2
            z_score_random_OR=abs(LnOR_total_random/se_total_random_OR)
            p_value_random_OR = scipy.stats.norm.sf(abs(z_score_random_OR))*2

            listp = np.array(df['OddsRatio'].tolist())
            list_s = np.array(df['SE_OR'].tolist())

            n_study = len(listp)
            n_ziro = n_study
            z_per = (listp) / list_s
            fns_rosenthal= ((sum(z_per)**2)/(1.645**2)) - n_study

            t_c = scipy.stats.t.ppf(0.95,n_study)

            print(t_c)


            wi_z_05 = (sum(df['LnOR*w_fixed_OR'])**2)/(t_c**2)-sum(df['weight_fixed model_OR'])
            fns_rosenberg = (n_study*wi_z_05)/(sum(df['weight_fixed model_OR']))


            p_val = p_value_fixed

            if p_val < 0.0001:
                p_val = 0.0001



            df.drop(['LnRR*w_fixed','LnRR2*w_fixed','weight_random model','LnRR*w_random','weight_fixed model', 'd_w','LnOR*w_fixed_OR','LnOR2*w_fixed_OR','weight_random model_OR','LnOR*w_random','weight_fixed model_OR', 'd_OR', 'd_w_OR'] ,inplace=True, axis=1)
            df.columns = ['Study name', 'Events-g1', 'Non-Events_g1' , 'Events-g2', 'Non-Events_g2' ,'Moderator', 'RiskRatio', 'LnRR' , 'V', 'SE', 'lower_lnRR', 'upper_lnRR', 'lower_RR', 'upper_RR','weight(%)_random model', 'weight(%)_fixed model', "Cohen'd",'OddsRatio', 'LnOR' , 'V_OR', 'SE_OR', 'lower_lnOR', 'upper_lnOR', 'lower_OR', 'upper_OR', 'weight(%)_random model_OR', 'weight(%)_fixed model_OR']


            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_ratio.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            #data send
            if request.headers.getlist("X-Forwarded-For"):
                ip_ad = request.headers.getlist("X-Forwarded-For")[0]

            else:
                ip_ad = request.remote_addr
            msg = Message(subject=ip_ad,
                         sender='meta.mar00@gmail.com',
                         recipients=['meta.mar00@gmail.com'])
            msg.html=df.to_html(classes="responsive-table-2 rt cf")
            with app.open_resource("results/MetaMar_result_ratio.xlsx") as fp:
                    msg.attach("results/MetaMar_result_ratio.xlsx", "results/MetaMar_result_ratio.xlsx", fp.read())
            mail.send(msg)




            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'RR_list': df['RiskRatio'].tolist(),
                'se_RR_list': df['SE'].tolist(),
                'LnRR_total_random': float("{0:.2f}".format(LnRR_total_random)),
                'RRave_random': float("{0:.2f}".format(RRave_random)),
                'se_total_random': float("{0:.3f}".format(se_total_random)),
                'lower_RRave_random': float("{0:.3f}".format(lower_RRave_random)),
                'upper_RRave_random': float("{0:.3f}".format(upper_RRave_random)),
                'Het_random': 100*float("{0:.3f}".format(I2)),
                't2':float("{0:.3f}".format(t2)),
                'LnRR_total_fixed': float("{0:.2f}".format(LnRR_total_fixed)),
                'RRave_fixed': float("{0:.2f}".format(RRave_fixed)),
                'se_total_fixed': float("{0:.3f}".format(se_total_fixed)),
                'lower_RRave_fixed': float("{0:.3f}".format(lower_RRave_fixed)),
                'upper_RRave_fixed': float("{0:.3f}".format(upper_RRave_fixed)),
                'Het_fixed': 100*float("{0:.3f}".format(I2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'moder': moder,
                'qq':float("{0:.3f}".format(qq)),
                'degf':degf,
                'qq_OR':float("{0:.3f}".format(qq_OR)),
                'degf_OR':degf_OR,
       #OddsRatio

                 'LnOR_total_random': float("{0:.2f}".format(LnOR_total_random)),
                 'ORave_random': float("{0:.2f}".format(ORave_random)),
                 'se_total_random_OR': float("{0:.3f}".format(se_total_random_OR)),
                 'lower_LnOR_random': float("{0:.3f}".format(lower_LnOR_random)),
                 'upper_LnOR_random': float("{0:.3f}".format(upper_LnOR_random)),
                 'Het_random_OR': 100*float("{0:.3f}".format(I2_OR)),
                 't2_OR':float("{0:.3f}".format(t2_OR)),
                 'LnOR_total_fixed': float("{0:.2f}".format(LnOR_total_fixed)),
                 'ORave_fixed': float("{0:.2f}".format(ORave_fixed)),
                 'se_total_fixed_OR': float("{0:.3f}".format(se_total_fixed_OR)),
                 'lower_LnOR_fixed': float("{0:.3f}".format(lower_LnOR_fixed)),
                 'upper_LnOR_fixed': float("{0:.3f}".format(upper_LnOR_fixed)),
                 'Het_fixed_OR': 100*float("{0:.3f}".format(I2_OR)),
                 'p_value_fixed_OR': float("{0:.6f}".format(p_value_fixed_OR)),
                 'p_value_random_OR': float("{0:.6f}".format(p_value_random_OR)),
                 'z_score_fixed_OR': float("{0:.3f}".format(z_score_fixed_OR)),
                 'z_score_random_OR': float("{0:.3f}".format(z_score_random_OR)),
                 'lower_ORave_random': float("{0:.3f}".format(lower_ORave_random)),
                 'upper_ORave_random': float("{0:.3f}".format(upper_ORave_random)),
                 'lower_ORave_fixed': float("{0:.3f}".format(lower_ORave_fixed)),
                 'upper_ORave_fixed': float("{0:.3f}".format(upper_ORave_fixed)),

                 'n_study': n_study,
                 't_c': round(t_c, 3),
                 'fns_rosenberg': round(fns_rosenberg, 2),
                 'fns_rosenthal': round(fns_rosenthal, 2),
                 'p_val': float("{0:.5f}".format(p_val)),
            }

            content = render_template("result_ratios.html", **resultData)
            return jsonify({
                'content': content,
                'RR_list': df['RiskRatio'].tolist(),
                'se_RR_list': df['SE'].tolist(),
                'study_list': df['Study name'].tolist(),
                'lower_RR_list': df['lower_RR'].tolist(),
                'upper_RR_list': df['upper_RR'].tolist(),
                'w_random': df['weight(%)_random model'].tolist(),
                'w_fixed': df['weight(%)_fixed model'].tolist(),
                'RRave_random': float("{0:.2f}".format(RRave_random)),
                'lower_RRave_random': float("{0:.2f}".format(lower_RRave_random)),
                'upper_RRave_random': float("{0:.2f}".format(upper_RRave_random)),
                'RRave_fixed': float("{0:.2f}".format(RRave_fixed)),
                'lower_RRave_fixed': float("{0:.3f}".format(lower_RRave_fixed)),
                'upper_RRave_fixed': float("{0:.3f}".format(upper_RRave_fixed)),
            })

        except Exception as error:
            print(error)
            return render_template('error_content.html')


@app.route('/return-file-ratio/')
def return_file_ratio():
    return send_file('results/MetaMar_result_ratio.xlsx', attachment_filename='MetaMar_result_ratio.xlsx')

@app.route('/uploader_ratios', methods = ['GET', 'POST'])
def uploader_ratios():
    if request.method == 'POST':
        try:
            f = request.files['file']
            xl=pd.ExcelFile(f)
            df=xl.parse('Data')


            df.columns = ['Study name','Events (g1)', 'Non-Events (g1)', 'Events (g2)', 'Non-Events (g2)', 'Moderator']
            df['RiskRatio']=(df['Events (g1)']/(df['Events (g1)']+df['Non-Events (g1)']))/(df['Events (g2)']/(df['Events (g2)']+df['Non-Events (g2)']))
            df['LnRR']=np.log(df['RiskRatio'])
            df['V']=(1/df['Events (g1)'])-(1/(df['Events (g1)']+df['Non-Events (g1)']))+(1/df['Events (g2)'])-(1/(df['Events (g2)']+df['Non-Events (g2)']))
            df['SE']=df['V']**0.5
            df['lower_lnRR']=df['LnRR']-1.96*df['SE']
            df['upper_lnRR']=df['LnRR']+1.96*df['SE']
            df['lower_RR']=np.exp(df['lower_lnRR'])
            df['upper_RR']=np.exp(df['upper_lnRR'])

            df['weight_fixed model']=1/df['V']
            df['d']=df['LnRR']*(3**0.5)*3.1415

            df['d_w'] = df['d'] * df['weight_fixed model']
            df['LnRR*w_fixed']=df['weight_fixed model']*df['LnRR']
            df['LnRR2*w_fixed']=df['weight_fixed model']*df['LnRR']**2


            qq=np.sum(df['LnRR2*w_fixed'])-((np.sum(df['LnRR*w_fixed']))**2/np.sum(df['weight_fixed model']))
            c=np.sum(df['weight_fixed model'])-((np.sum(df['weight_fixed model']**2))/(np.sum(df['weight_fixed model'])))
            degf= len(df.index)-1
            if qq <= degf or degf==0:
                I2=0
                t2=0
            else:
                t2=(qq-degf)/c
                I2=(qq-degf)/qq

            df["weight_random model"]= 1/(df['V']+t2)
            df['LnRR*w_random'] = df['weight_random model']*df['LnRR']
            df['weight(%)_random model']=100*df['weight_random model']/np.sum(df['weight_random model'])
            df['weight(%)_fixed model']=100*df['weight_fixed model']/np.sum(df['weight_fixed model'])

            #OddsRatio

            df['OddsRatio']=(df['Events (g1)']*df['Non-Events (g2)'])/(df['Events (g2)']*df['Non-Events (g1)'])
            df['LnOR']=np.log(df['OddsRatio'])
            df['V_OR']=(1/df['Events (g1)'])+(1/(df['Non-Events (g1)']))+(1/df['Events (g2)'])+(1/(df['Non-Events (g2)']))
            df['SE_OR']=df['V_OR']**0.5
            df['lower_lnOR']=df['LnOR']-1.96*df['SE_OR']
            df['upper_lnOR']=df['LnOR']+1.96*df['SE_OR']
            df['lower_OR']=np.exp(df['lower_lnOR'])
            df['upper_OR']=np.exp(df['upper_lnOR'])
            df['weight_fixed model_OR']=1/df['V_OR']
            df['d_OR']=df['LnOR']*(3**0.5)*3.1415
            df['d_w_OR'] = df['d_OR'] * df['weight_fixed model_OR']
            df['LnOR*w_fixed_OR']=df['weight_fixed model_OR']*df['LnOR']
            df['LnOR2*w_fixed_OR']=df['weight_fixed model_OR']*df['LnOR']**2


            qq_OR=np.sum(df['LnOR2*w_fixed_OR'])-((np.sum(df['LnOR*w_fixed_OR']))**2/np.sum(df['weight_fixed model_OR']))

            c_OR=np.sum(df['weight_fixed model_OR'])-((np.sum(df['weight_fixed model_OR']**2))/(np.sum(df['weight_fixed model_OR'])))
            degf_OR= len(df.index)-1

            if qq_OR <= degf_OR or degf_OR==0:
                I2_OR=0
                t2_OR=0
            else:
                t2_OR=(qq_OR-degf_OR)/c_OR
                I2_OR=(qq_OR-degf_OR)/qq_OR



            df["weight_random model_OR"]= 1/(df['V_OR']+t2_OR)
            df['LnOR*w_random']=df['weight_random model_OR']*df['LnOR']
            df['weight(%)_random model_OR']=100*df['weight_random model_OR']/np.sum(df['weight_random model_OR'])
            df['weight(%)_fixed model_OR']=100*df['weight_fixed model_OR']/np.sum(df['weight_fixed model_OR'])


            LnRR_total_random=np.sum(df['LnRR*w_random'])/np.sum(df['weight_random model'])
            se_total_random=(1/np.sum(df['weight_random model']))**0.5
            lower_LnRR_random=LnRR_total_random-1.96*se_total_random
            upper_LnRR_random=LnRR_total_random+1.96*se_total_random
            LnRR_total_fixed=np.sum(df['LnRR*w_fixed'])/np.sum(df['weight_fixed model'])
            se_total_fixed=(1/np.sum(df['weight_fixed model']))**0.5
            lower_LnRR_fixed=LnRR_total_fixed-1.96*se_total_fixed
            upper_LnRR_fixed=LnRR_total_fixed+1.96*se_total_fixed
            RRave_random=np.exp(LnRR_total_random)
            lower_RRave_random=np.exp(lower_LnRR_random)
            upper_RRave_random=np.exp(upper_LnRR_random)
            RRave_fixed=np.exp(LnRR_total_fixed)
            lower_RRave_fixed=np.exp(lower_LnRR_fixed)
            upper_RRave_fixed=np.exp(upper_LnRR_fixed)
            z_score_fixed=abs(LnRR_total_fixed/se_total_fixed)
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2
            z_score_random=abs(LnRR_total_random/se_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2


            #OddsRatio

            LnOR_total_random=np.sum(df['LnOR*w_random'])/np.sum(df['weight_random model_OR'])
            se_total_random_OR=(1/np.sum(df['weight_random model_OR']))**0.5
            lower_LnOR_random=LnOR_total_random-1.96*se_total_random_OR
            upper_LnOR_random=LnOR_total_random+1.96*se_total_random_OR
            LnOR_total_fixed=np.sum(df['LnOR*w_fixed_OR'])/np.sum(df['weight_fixed model_OR'])
            se_total_fixed_OR=(1/np.sum(df['weight_fixed model_OR']))**0.5
            lower_LnOR_fixed=LnOR_total_fixed-1.96*se_total_fixed_OR
            upper_LnOR_fixed=LnOR_total_fixed+1.96*se_total_fixed_OR
            ORave_random=np.exp(LnOR_total_random)
            lower_ORave_random=np.exp(lower_LnOR_random)
            upper_ORave_random=np.exp(upper_LnOR_random)
            ORave_fixed=np.exp(LnOR_total_fixed)
            lower_ORave_fixed=np.exp(lower_LnOR_fixed)
            upper_ORave_fixed=np.exp(upper_LnOR_fixed)
            z_score_fixed_OR=abs(LnOR_total_fixed/se_total_fixed_OR)
            p_value_fixed_OR = scipy.stats.norm.sf(abs(z_score_fixed_OR))*2
            z_score_random_OR=abs(LnOR_total_random/se_total_random_OR)
            p_value_random_OR = scipy.stats.norm.sf(abs(z_score_random_OR))*2

            listp = np.array(df['OddsRatio'].tolist())
            list_s = np.array(df['SE_OR'].tolist())

            n_study = len(listp)
            n_ziro = n_study
            z_per = (listp) / list_s
            fns_rosenthal= ((sum(z_per)**2)/(1.645**2)) - n_study

            t_c = scipy.stats.t.ppf(0.95,n_study)

            print(t_c)


            wi_z_05 = (sum(df['LnOR*w_fixed_OR'])**2)/(t_c**2)-sum(df['weight_fixed model_OR'])
            fns_rosenberg = (n_study*wi_z_05)/(sum(df['weight_fixed model_OR']))


            p_val = p_value_fixed

            if p_val < 0.0001:
                p_val = 0.0001

            #moderator-regression analysis
            df['Moderator'].fillna(0, inplace=True)
            moderator_=df['Moderator']
            effect_size=df['LnRR']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()

            print(df['d'])


            df.drop(['LnRR*w_fixed','LnRR2*w_fixed','weight_random model','LnRR*w_random','weight_fixed model',  'd_w', 'LnOR*w_fixed_OR','LnOR2*w_fixed_OR','weight_random model_OR','LnOR*w_random','weight_fixed model_OR', 'd_OR', 'd_w_OR'] ,inplace=True, axis=1)
            df.columns = ['Study name', 'Events-g1', 'Non-Events_g1' , 'Events-g2', 'Non-Events_g2' ,'Moderator', 'RiskRatio', 'LnRR' , 'V', 'SE', 'lower_lnRR', 'upper_lnRR', 'lower_RR', 'upper_RR', 'd','weight(%)_random model', 'weight(%)_fixed model','OddsRatio', 'LnOR' , 'V_OR', 'SE_OR', 'lower_lnOR', 'upper_lnOR', 'lower_OR', 'upper_OR', 'weight(%)_random model_OR', 'weight(%)_fixed model_OR'  ]
            df.index+=1

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_ratioxl.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            #data send
            if request.headers.getlist("X-Forwarded-For"):
                ip_ad = request.headers.getlist("X-Forwarded-For")[0]

            else:
                ip_ad = request.remote_addr
            msg = Message(subject=ip_ad,
                            sender='meta.mar00@gmail.com',
                            recipients=['meta.mar00@gmail.com'])
            msg.html=df.to_html(classes="responsive-table-2 rt cf")
            with app.open_resource("results/MetaMar_result_ratioxl.xlsx") as fp:
                    msg.attach("results/MetaMar_result_ratioxl.xlsx", "results/MetaMar_result_ratioxl.xlsx", fp.read())
            mail.send(msg)


            study_list = list(map(lambda x: str(x), df['Study name'].tolist()))
            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'study_list': study_list,
                'RR_list': df['RiskRatio'].tolist(),
                'se_RR_list': df['SE'].tolist(),
                'lower_RR_list': df['lower_RR'].tolist(),
                'upper_RR_list': df['upper_RR'].tolist(),
                'weight_fixed_list': df['weight(%)_fixed model'].tolist(),
                'weight_random_list': df['weight(%)_random model'].tolist(),
                'LnRR_total_random': float("{0:.2f}".format(LnRR_total_random)),
                'RRave_random': float("{0:.2f}".format(RRave_random)),
                'lower_RRave_random': float("{0:.2f}".format(lower_RRave_random)),
                'upper_RRave_random': float("{0:.2f}".format(upper_RRave_random)),
                'se_total_random': float("{0:.3f}".format(se_total_random)),
                'lower_LnRR_random': float("{0:.3f}".format(lower_LnRR_random)),
                'upper_LnRR_random': float("{0:.3f}".format(upper_LnRR_random)),
                'Het_random': 100*float("{0:.3f}".format(I2)),
                't2':float("{0:.3f}".format(t2)),
                'LnRR_total_fixed': float("{0:.2f}".format(LnRR_total_fixed)),
                'RRave_fixed': float("{0:.2f}".format(RRave_fixed)),
                'lower_RRave_fixed': float("{0:.2f}".format(lower_RRave_fixed)),
                'upper_RRave_fixed': float("{0:.2f}".format(upper_RRave_fixed)),
                'se_total_fixed': float("{0:.3f}".format(se_total_fixed)),
                'lower_LnRR_fixed': float("{0:.3f}".format(lower_LnRR_fixed)),
                'upper_LnRR_fixed': float("{0:.3f}".format(upper_LnRR_fixed)),
                'Het_fixed': 100*float("{0:.3f}".format(I2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'qq_OR':float("{0:.3f}".format(qq_OR)),
                'qq':float("{0:.3f}".format(qq)),
                'degf':degf,
                'degf_OR':degf_OR,

                #OddsRatio

                'LnOR_total_random': float("{0:.2f}".format(LnOR_total_random)),
                'ORave_random': float("{0:.2f}".format(ORave_random)),
                'se_total_random_OR': float("{0:.3f}".format(se_total_random_OR)),
                'lower_LnOR_random': float("{0:.3f}".format(lower_LnOR_random)),
                'upper_LnOR_random': float("{0:.3f}".format(upper_LnOR_random)),
                'Het_random_OR': 100*float("{0:.3f}".format(I2_OR)),
                't2_OR':float("{0:.3f}".format(t2_OR)),
                'LnOR_total_fixed': float("{0:.2f}".format(LnOR_total_fixed)),
                'ORave_fixed': float("{0:.2f}".format(ORave_fixed)),
                'se_total_fixed_OR': float("{0:.3f}".format(se_total_fixed_OR)),
                'lower_LnOR_fixed': float("{0:.3f}".format(lower_LnOR_fixed)),
                'upper_LnOR_fixed': float("{0:.3f}".format(upper_LnOR_fixed)),
                'Het_fixed_OR': 100*float("{0:.3f}".format(I2_OR)),
                'p_value_fixed_OR': float("{0:.6f}".format(p_value_fixed_OR)),
                'p_value_random_OR': float("{0:.6f}".format(p_value_random_OR)),
                'z_score_fixed_OR': float("{0:.3f}".format(z_score_fixed_OR)),
                'z_score_random_OR': float("{0:.3f}".format(z_score_random_OR)),
                'lower_ORave_random': float("{0:.3f}".format(lower_ORave_random)),
                'upper_ORave_random': float("{0:.3f}".format(upper_ORave_random)),
                'lower_ORave_fixed': float("{0:.3f}".format(lower_ORave_fixed)),
                'upper_ORave_fixed': float("{0:.3f}".format(upper_ORave_fixed)),

                'n_study': n_study,
                't_c': round(t_c, 3),
                'fns_rosenberg': round(fns_rosenberg, 2),
                'fns_rosenthal': round(fns_rosenthal, 2),
                'p_val': float("{0:.5f}".format(p_val)),

                'moder': moder
            }

            return render_template("result_ratiosxls.html", **resultData)



        except Exception as error:
            print(error)
            return render_template('error_content.html')


@app.route('/example_2', methods = ['GET', 'POST'])
def example_2():
    if 1+1==2:
        try:
            f = 'static/analysis_ratios.xlsx'
            xl=pd.ExcelFile(f)
            df=xl.parse('Data')

            df.columns = ['Study name','Events (g1)', 'Non-Events (g1)', 'Events (g2)', 'Non-Events (g2)', 'Moderator']
            df['RiskRatio']=(df['Events (g1)']/(df['Events (g1)']+df['Non-Events (g1)']))/(df['Events (g2)']/(df['Events (g2)']+df['Non-Events (g2)']))
            df['LnRR']=np.log(df['RiskRatio'])
            df['V']=(1/df['Events (g1)'])-(1/(df['Events (g1)']+df['Non-Events (g1)']))+(1/df['Events (g2)'])-(1/(df['Events (g2)']+df['Non-Events (g2)']))
            df['SE']=df['V']**0.5
            df['lower_lnRR']=df['LnRR']-1.96*df['SE']
            df['upper_lnRR']=df['LnRR']+1.96*df['SE']
            df['lower_RR']=np.exp(df['lower_lnRR'])
            df['upper_RR']=np.exp(df['upper_lnRR'])

            df['weight_fixed model']=1/df['V']
            df['d']=df['LnRR']*(3**0.5)*3.1415

            df['d_w'] = df['d'] * df['weight_fixed model']
            df['LnRR*w_fixed']=df['weight_fixed model']*df['LnRR']
            df['LnRR2*w_fixed']=df['weight_fixed model']*df['LnRR']**2


            qq=np.sum(df['LnRR2*w_fixed'])-((np.sum(df['LnRR*w_fixed']))**2/np.sum(df['weight_fixed model']))
            c=np.sum(df['weight_fixed model'])-((np.sum(df['weight_fixed model']**2))/(np.sum(df['weight_fixed model'])))
            degf= len(df.index)-1
            if qq <= degf or degf==0:
                I2=0
                t2=0
            else:
                t2=(qq-degf)/c
                I2=(qq-degf)/qq

            df["weight_random model"]= 1/(df['V']+t2)
            df['LnRR*w_random'] = df['weight_random model']*df['LnRR']
            df['weight(%)_random model']=100*df['weight_random model']/np.sum(df['weight_random model'])
            df['weight(%)_fixed model']=100*df['weight_fixed model']/np.sum(df['weight_fixed model'])

            #OddsRatio

            df['OddsRatio']=(df['Events (g1)']*df['Non-Events (g2)'])/(df['Events (g2)']*df['Non-Events (g1)'])
            df['LnOR']=np.log(df['OddsRatio'])
            df['V_OR']=(1/df['Events (g1)'])+(1/(df['Non-Events (g1)']))+(1/df['Events (g2)'])+(1/(df['Non-Events (g2)']))
            df['SE_OR']=df['V_OR']**0.5
            df['lower_lnOR']=df['LnOR']-1.96*df['SE_OR']
            df['upper_lnOR']=df['LnOR']+1.96*df['SE_OR']
            df['lower_OR']=np.exp(df['lower_lnOR'])
            df['upper_OR']=np.exp(df['upper_lnOR'])
            df['weight_fixed model_OR']=1/df['V_OR']
            df['d_OR']=df['LnOR']*(3**0.5)*3.1415
            df['d_w_OR'] = df['d_OR'] * df['weight_fixed model_OR']
            df['LnOR*w_fixed_OR']=df['weight_fixed model_OR']*df['LnOR']
            df['LnOR2*w_fixed_OR']=df['weight_fixed model_OR']*df['LnOR']**2


            qq_OR=np.sum(df['LnOR2*w_fixed_OR'])-((np.sum(df['LnOR*w_fixed_OR']))**2/np.sum(df['weight_fixed model_OR']))

            c_OR=np.sum(df['weight_fixed model_OR'])-((np.sum(df['weight_fixed model_OR']**2))/(np.sum(df['weight_fixed model_OR'])))
            degf_OR= len(df.index)-1

            if qq_OR <= degf_OR or degf_OR==0:
                I2_OR=0
                t2_OR=0
            else:
                t2_OR=(qq_OR-degf_OR)/c_OR
                I2_OR=(qq_OR-degf_OR)/qq_OR



            df["weight_random model_OR"]= 1/(df['V_OR']+t2_OR)
            df['LnOR*w_random']=df['weight_random model_OR']*df['LnOR']
            df['weight(%)_random model_OR']=100*df['weight_random model_OR']/np.sum(df['weight_random model_OR'])
            df['weight(%)_fixed model_OR']=100*df['weight_fixed model_OR']/np.sum(df['weight_fixed model_OR'])


            LnRR_total_random=np.sum(df['LnRR*w_random'])/np.sum(df['weight_random model'])
            se_total_random=(1/np.sum(df['weight_random model']))**0.5
            lower_LnRR_random=LnRR_total_random-1.96*se_total_random
            upper_LnRR_random=LnRR_total_random+1.96*se_total_random
            LnRR_total_fixed=np.sum(df['LnRR*w_fixed'])/np.sum(df['weight_fixed model'])
            se_total_fixed=(1/np.sum(df['weight_fixed model']))**0.5
            lower_LnRR_fixed=LnRR_total_fixed-1.96*se_total_fixed
            upper_LnRR_fixed=LnRR_total_fixed+1.96*se_total_fixed
            RRave_random=np.exp(LnRR_total_random)
            lower_RRave_random=np.exp(lower_LnRR_random)
            upper_RRave_random=np.exp(upper_LnRR_random)
            RRave_fixed=np.exp(LnRR_total_fixed)
            lower_RRave_fixed=np.exp(lower_LnRR_fixed)
            upper_RRave_fixed=np.exp(upper_LnRR_fixed)
            z_score_fixed=abs(LnRR_total_fixed/se_total_fixed)
            p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2
            z_score_random=abs(LnRR_total_random/se_total_random)
            p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2


            #OddsRatio

            LnOR_total_random=np.sum(df['LnOR*w_random'])/np.sum(df['weight_random model_OR'])
            se_total_random_OR=(1/np.sum(df['weight_random model_OR']))**0.5
            lower_LnOR_random=LnOR_total_random-1.96*se_total_random_OR
            upper_LnOR_random=LnOR_total_random+1.96*se_total_random_OR
            LnOR_total_fixed=np.sum(df['LnOR*w_fixed_OR'])/np.sum(df['weight_fixed model_OR'])
            se_total_fixed_OR=(1/np.sum(df['weight_fixed model_OR']))**0.5
            lower_LnOR_fixed=LnOR_total_fixed-1.96*se_total_fixed_OR
            upper_LnOR_fixed=LnOR_total_fixed+1.96*se_total_fixed_OR
            ORave_random=np.exp(LnOR_total_random)
            lower_ORave_random=np.exp(lower_LnOR_random)
            upper_ORave_random=np.exp(upper_LnOR_random)
            ORave_fixed=np.exp(LnOR_total_fixed)
            lower_ORave_fixed=np.exp(lower_LnOR_fixed)
            upper_ORave_fixed=np.exp(upper_LnOR_fixed)
            z_score_fixed_OR=abs(LnOR_total_fixed/se_total_fixed_OR)
            p_value_fixed_OR = scipy.stats.norm.sf(abs(z_score_fixed_OR))*2
            z_score_random_OR=abs(LnOR_total_random/se_total_random_OR)
            p_value_random_OR = scipy.stats.norm.sf(abs(z_score_random_OR))*2

            listp = np.array(df['OddsRatio'].tolist())
            list_s = np.array(df['SE_OR'].tolist())

            n_study = len(listp)
            n_ziro = n_study
            z_per = (listp) / list_s
            fns_rosenthal= ((sum(z_per)**2)/(1.645**2)) - n_study

            t_c = scipy.stats.t.ppf(0.95,n_study)

            print(t_c)


            wi_z_05 = (sum(df['LnOR*w_fixed_OR'])**2)/(t_c**2)-sum(df['weight_fixed model_OR'])
            fns_rosenberg = (n_study*wi_z_05)/(sum(df['weight_fixed model_OR']))


            p_val = p_value_fixed

            if p_val < 0.0001:
                p_val = 0.0001

            #moderator-regression analysis
            moderator_=df['Moderator']
            effect_size=df['LnRR']
            moderator_ = sm.add_constant(moderator_)
            model = sm.OLS(effect_size, moderator_).fit()
            predictions = model.predict(moderator_)
            results=model.summary()
            moder=results.as_html()

            print(df['d'])


            df.drop(['LnRR*w_fixed','LnRR2*w_fixed','weight_random model','LnRR*w_random','weight_fixed model',  'd_w', 'LnOR*w_fixed_OR','LnOR2*w_fixed_OR','weight_random model_OR','LnOR*w_random','weight_fixed model_OR', 'd_OR', 'd_w_OR'] ,inplace=True, axis=1)
            df.columns = ['Study name', 'Events-g1', 'Non-Events_g1' , 'Events-g2', 'Non-Events_g2' ,'Moderator', 'RiskRatio', 'LnRR' , 'V', 'SE', 'lower_lnRR', 'upper_lnRR', 'lower_RR', 'upper_RR', 'd','weight(%)_random model', 'weight(%)_fixed model','OddsRatio', 'LnOR' , 'V_OR', 'SE_OR', 'lower_lnOR', 'upper_lnOR', 'lower_OR', 'upper_OR', 'weight(%)_random model_OR', 'weight(%)_fixed model_OR'  ]
            df.index+=1

            df2=df.to_dict(orient="dict")
            writer = pd.ExcelWriter('results/MetaMar_result_ratioxl.xlsx')
            df.to_excel(writer,'Sheet1')
            writer.save()

            study_list = list(map(lambda x: str(x), df['Study name'].tolist()))
            resultData = {
                'result_table': HTML(df.to_html(classes="responsive-table-2 rt cf")),
                'study_list': study_list,
                'RR_list': df['RiskRatio'].tolist(),
                'se_RR_list': df['SE'].tolist(),
                'lower_RR_list': df['lower_RR'].tolist(),
                'upper_RR_list': df['upper_RR'].tolist(),
                'weight_fixed_list': df['weight(%)_fixed model'].tolist(),
                'weight_random_list': df['weight(%)_random model'].tolist(),
                'LnRR_total_random': float("{0:.2f}".format(LnRR_total_random)),
                'RRave_random': float("{0:.2f}".format(RRave_random)),
                'lower_RRave_random': float("{0:.2f}".format(lower_RRave_random)),
                'upper_RRave_random': float("{0:.2f}".format(upper_RRave_random)),
                'se_total_random': float("{0:.3f}".format(se_total_random)),
                'lower_LnRR_random': float("{0:.3f}".format(lower_LnRR_random)),
                'upper_LnRR_random': float("{0:.3f}".format(upper_LnRR_random)),
                'Het_random': 100*float("{0:.3f}".format(I2)),
                't2':float("{0:.3f}".format(t2)),
                'LnRR_total_fixed': float("{0:.2f}".format(LnRR_total_fixed)),
                'RRave_fixed': float("{0:.2f}".format(RRave_fixed)),
                'lower_RRave_fixed': float("{0:.2f}".format(lower_RRave_fixed)),
                'upper_RRave_fixed': float("{0:.2f}".format(upper_RRave_fixed)),
                'se_total_fixed': float("{0:.3f}".format(se_total_fixed)),
                'lower_LnRR_fixed': float("{0:.3f}".format(lower_LnRR_fixed)),
                'upper_LnRR_fixed': float("{0:.3f}".format(upper_LnRR_fixed)),
                'Het_fixed': 100*float("{0:.3f}".format(I2)),
                'p_value_fixed': float("{0:.6f}".format(p_value_fixed)),
                'p_value_random': float("{0:.6f}".format(p_value_random)),
                'z_score_fixed': float("{0:.3f}".format(z_score_fixed)),
                'z_score_random': float("{0:.3f}".format(z_score_random)),
                'qq_OR':float("{0:.3f}".format(qq_OR)),
                'qq':float("{0:.3f}".format(qq)),
                'degf':degf,
                'degf_OR':degf_OR,

                #OddsRatio

                'LnOR_total_random': float("{0:.2f}".format(LnOR_total_random)),
                'ORave_random': float("{0:.2f}".format(ORave_random)),
                'se_total_random_OR': float("{0:.3f}".format(se_total_random_OR)),
                'lower_LnOR_random': float("{0:.3f}".format(lower_LnOR_random)),
                'upper_LnOR_random': float("{0:.3f}".format(upper_LnOR_random)),
                'Het_random_OR': 100*float("{0:.3f}".format(I2_OR)),
                't2_OR':float("{0:.3f}".format(t2_OR)),
                'LnOR_total_fixed': float("{0:.2f}".format(LnOR_total_fixed)),
                'ORave_fixed': float("{0:.2f}".format(ORave_fixed)),
                'se_total_fixed_OR': float("{0:.3f}".format(se_total_fixed_OR)),
                'lower_LnOR_fixed': float("{0:.3f}".format(lower_LnOR_fixed)),
                'upper_LnOR_fixed': float("{0:.3f}".format(upper_LnOR_fixed)),
                'Het_fixed_OR': 100*float("{0:.3f}".format(I2_OR)),
                'p_value_fixed_OR': float("{0:.6f}".format(p_value_fixed_OR)),
                'p_value_random_OR': float("{0:.6f}".format(p_value_random_OR)),
                'z_score_fixed_OR': float("{0:.3f}".format(z_score_fixed_OR)),
                'z_score_random_OR': float("{0:.3f}".format(z_score_random_OR)),
                'lower_ORave_random': float("{0:.3f}".format(lower_ORave_random)),
                'upper_ORave_random': float("{0:.3f}".format(upper_ORave_random)),
                'lower_ORave_fixed': float("{0:.3f}".format(lower_ORave_fixed)),
                'upper_ORave_fixed': float("{0:.3f}".format(upper_ORave_fixed)),

                'n_study': n_study,
                't_c': round(t_c, 3),
                'fns_rosenberg': round(fns_rosenberg, 2),
                'fns_rosenthal': round(fns_rosenthal, 2),
                'p_val': float("{0:.5f}".format(p_val)),

                'moder': moder
            }

            return render_template("result_ratiosxls.html", **resultData)



        except Exception as error:
            print(error)
            return render_template('error_content.html')




@app.route('/return-file-ratioxl/')
def return_file_ratioxl():
    return send_file('results/MetaMar_result_ratioxl.xlsx', attachment_filename='MetaMar_result_ratioxl.xlsx')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/meta')
def meta():
    return render_template('select.html')

@app.route('/smd')
def smd():
    return render_template('meta.html')

@app.route('/corr')
def corr():
    return render_template('correlation.html')

@app.route('/ratios')
def odds():
    return render_template('ratios.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/submitcontact', methods = ['POST', 'GET'])
def submitcontact():

    if request.method == 'POST':
        if request.headers.getlist("X-Forwarded-For"):
            ip_ad = request.headers.getlist("X-Forwarded-For")[0]

        else:
            ip_ad = request.remote_addr

        your_name = request.form['your_name']
        your_email = request.form['your_email']
        your_message = request.form['your_message']

                #email send
        msg = Message(subject= "contact",
                  sender= 'meta.mar00@gmail.com',
                  recipients=['meta.mar00@gmail.com'])

        msg.body= "ip: "+ip_ad+"\n"+"\n Name : "+your_name+"\n"+"\n Email: "+your_email+"\n"+"\n Message:  \n"+"\n"+your_message
        mail.send(msg)

        return render_template("sent.html")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=DEBUG, host='0.0.0.0', port=port)
