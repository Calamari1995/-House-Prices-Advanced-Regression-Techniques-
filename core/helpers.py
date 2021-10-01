import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from jupyterthemes import jtplot

rcdict = {
    'grid.linewidth'	: 1,
    'xtick.major.size'	: 0,
    'ytick.major.size'	: 0,
    'xtick.minor.size'	: 0,
    'ytick.minor.size'	: 0 ,
    'axes.labelpad'		: 10
}

jtplot.style(theme='grade3', context='talk', fscale=1.4, spines=False, gridlines='--')
sns.set_context('talk', rc=rcdict)

# ------------------------- #

def true_pred_clouds(

		model,
		X_train,
		X_test,
		y_train,
		y_test,
		figsize: tuple = (14, 10),
		**kwargs

	):

	

	fig, ax = plt.subplots(figsize=figsize)
	
	pred = model.predict(X_train)
	predt = model.predict(X_test)

	if len(pred.shape) > 1: 

		pred = pred[:, 0]
		predt = predt[:, 0]

	sns.scatterplot(

		x=y_train, 
		y=pred,
		color='darkblue', 
		label='train',
		**kwargs

	)

	sns.scatterplot(

		x=y_test, 
		y=predt,
		color='crimson',
		label='val',
		**kwargs

	)

	ideal = [

		min(y_train.min(), y_test.min()), 
		max(y_train.max(), y_test.max())

	]

	sns.lineplot(

		x=ideal, 
		y=ideal, 
		linestyle='dashed', 
		alpha=0.6, 
		color='black', 
		label='ideal'

	)

	ax.set_xlabel(

		'True', 
		fontsize=16, 
		labelpad=15

	)

	ax.set_ylabel(

		'Predict', 
		fontsize=16, 
		labelpad=15

	)
	ax.set_title(

		'', 
		fontsize=20, 
		pad=15

	)

	ax.legend(fontsize=12)
	#ax.set_xlim(50, 350)
	#ax.set_ylim(50, 350)

	ax.set_title(

		f'train RMSE: {mean_squared_error(y_train, pred)**0.5:.3f} | ' \
		f'test RMSE: {mean_squared_error(y_test, predt)**0.5:.3f}', 
		fontsize=20, 
		pad=15

	)


	return fig, ax
