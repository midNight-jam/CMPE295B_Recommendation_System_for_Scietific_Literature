import datetime


right_now = str(datetime.datetime.now().isoformat())
right_now = right_now.replace(":","_")
right_now = right_now.replace(".","_")
right_now = right_now.replace("-","_")

# right_now = '2018_04_21T17_02_21_754258'

numbers_users = 5551
numbers_papers = 6001
number_words = 8000
number_latent_factors = 2500
top_k_products = 35



train_file = 'Data/trim_users_5551_6000_papers__test_0.3_.dat'
test_file = 'Data/trim_users_5551_6000_papers__train_0.3_.dat'
output_dir = "zzOutput/"
autoencoders_output_file = output_dir + "autoencoders__" + right_now
autoencoders_training_loss_readings_file = output_dir + "autoencoders__training_loss_" + right_now + ".dat"

precision_output_file = output_dir + "precision__" + right_now + ".dat"
precision_readings_output_file = output_dir + "precision__readings__" + right_now + ".dat"


precision_M_output_file = output_dir + "precision__M_" + right_now + ".dat"
precision_M_readings_output_file = output_dir + "precision__M_readings__" + right_now + ".dat"

recall_output_file = output_dir + "recall__" + right_now + ".dat"
recall_readings_output_file = output_dir + "recall__readings__" + right_now + ".dat"

svd_output_file = output_dir + "svd__" + right_now + ".dat"
sample_svd_output_file = output_dir + "svd__sample__" + right_now + ".dat"
trimmed_users_count = 5551
trimmed_papers_count = 6001

rating_threshold = 0.0
lib_size_threshold = 10
test_train_split = 0.30



epoch = 10
batch_size = 512

# evaluations config
M = top_k_products


def dump():
    print('~~' * 20)
    print(' -' * 20)
    print('rightnow   .......   {0}'.format(right_now))
    print('numbers_users   .......   {0}'.format(numbers_users))
    print('numbers_papers  .......   {0}'.format(numbers_papers))
    print('number_words  .......   {0}'.format(number_words))
    print('number_latent_factors  .......   {0}'.format(number_latent_factors))
    print('top_k_products  .......   {0}'.format(top_k_products))

    print('train_file  .......   {0}'.format(train_file))
    print('test_file  .......   {0}'.format(test_file))
    print('output_dir   .......   {0}'.format(output_dir))
    print('precision_output_file    .......   {0}'.format(precision_output_file))
    print('precision_readings_output_file    .......   {0}'.format(precision_readings_output_file))
    print('svd_output_file    .......   {0}'.format(svd_output_file))

    print('trimmed_users_count    .......   {0}'.format(trimmed_users_count))
    print('trimmed_papers_count    .......   {0}'.format(trimmed_papers_count))

    print('rating_threshold   .......   {0}'.format(rating_threshold))
    print('lib_size_threshold    .......   {0}'.format(lib_size_threshold))
    print('test_train_split    .......   {0}'.format(test_train_split))


    print('epoch   .......   {0}'.format(epoch))
    print('batch_size   .......   {0}'.format(batch_size))
    print(' -' * 20)
    print('~~' * 20)


dump()