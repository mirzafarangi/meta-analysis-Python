import numpy
import pandas
import scipy.stats
import statsmodels.api as statsModelsAPI
from flask import render_template, jsonify
from IPython.display import HTML


def calculate(input_data):
    try:
        study = [row['study'] for row in input_data]
        g1_sample = [row['g1_sample'] for row in input_data]
        g1_mean = [row['g1_mean'] for row in input_data]
        g1_sd = [row['g1_sd'] for row in input_data]

        g2_sample = [row['g2_sample'] for row in input_data]
        g2_mean = [row['g2_mean'] for row in input_data]
        g2_sd = [row['g2_sd'] for row in input_data]

        try:
            moderator = [row['moderator'] for row in input_data]
        except:
            moderator = ['moderator']

        table = [study, g1_sample, g1_mean, g1_sd, g2_sample, g2_mean, g2_sd, moderator]
        df = pandas.DataFrame(table)
        df = df.transpose()

        df.index+=1

        df = df.convert_objects(convert_numeric=True)

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

        g_total_fixed=numpy.sum(df['wg_fixed'])/numpy.sum(df['w_fixed'])
        se_total_fixed=(1/numpy.sum(df['w_fixed']))**0.5

        lower_g_fixed=g_total_fixed-1.96*se_total_fixed
        upper_g_fixed=g_total_fixed+1.96*se_total_fixed

        qq=numpy.sum(df['wg2_fixed'])-((numpy.sum(df['wg_fixed']))**2/numpy.sum(df['w_fixed']))
        c=numpy.sum(df['w_fixed'])-((numpy.sum(df['w_fixed']**2))/(numpy.sum(df['w_fixed'])))
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

        g_total_random=numpy.sum(df['wg_random'])/numpy.sum(df['w_random'])
        se_total_random=(1/numpy.sum(df['w_random']))**0.5

        lower_g_random=g_total_random-1.96*se_total_random
        upper_g_random=g_total_random+1.96*se_total_random

        df["weight(%)-fixed model"]=(df['w_fixed']/sum(df['w_fixed']))*100
        df["weight(%)-random model %"]=(df['w_random']/sum(df['w_random']))*100

        z_score_fixed=g_total_fixed/se_total_fixed
        p_value_fixed = scipy.stats.norm.sf(abs(z_score_fixed))*2

        z_score_random=g_total_random/se_total_random
        p_value_random = scipy.stats.norm.sf(abs(z_score_random))*2

        #Heterogeneity
        #random model
        I2_random=I2_fixed
        #moderator-regression analysis
        moderator_=df[7]
        effect_size=df['g']
        moderator_ = statsModelsAPI.add_constant(moderator_)
        model = statsModelsAPI.OLS(effect_size, moderator_).fit()
        # predictions = model.predict(moderator_)
        results=model.summary()
        moder=results.as_html()


        df.drop(['Spooled','d','n','n_1','SEd','d_lower','d_upper','w_d','wd','w_fixed','wg_fixed', 'wg2_fixed', 'w_random', 'wg_random', 'wg2_random'] ,inplace=True, axis=1)
        df.columns = ['Study name', 'n1', 'Mean1' , 'SD1', 'n2', 'Mean2' , 'SD2', 'Moderator Variable', 'g', 'SEg', 'g_lower', 'g_upper', 'weight(%)-fixed model', 'weight(%)-random model' ]
        # df2=df.to_dict(orient="dict")
        writer = pandas.ExcelWriter('results/MetaMar_result_smd.xlsx')
        df.to_excel(writer,'Sheet1')
        writer.save()

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
            'moder': moder
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
