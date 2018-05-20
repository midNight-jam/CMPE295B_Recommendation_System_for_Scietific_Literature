import config

def read_predictions_and_test():
  pred_user_dict = {}
  user_id = 0;

  for line in open(config.svd_output_file):
    pred_user_dict[user_id] = []
    pred = line.split()
    pred.pop(0)	# removing the length of predictions
    pred = map(int, pred)
    for p in pred:
        pred_user_dict[user_id].append(p)
    user_id += 1

  user_id = 0;
  test_user_dict = {}

  for line in open(config.test_file):
    docs = line.split()
    docs.pop(0)
    user_papers = map(int, docs)
    test_user_dict[user_id] = []
    for d in user_papers:
      test_user_dict[user_id].append(d)
    user_id += 1

  return pred_user_dict, test_user_dict


def precision(pred, test):
  if (len(pred) != len(test)):
    print("Lengths of predicted & test users dictionary doesnt match by row Count")
    return

  user_count = len(pred)
  final_precision = 0.0
  max_precision = 0.0
  included_users = 0

  precision_file = open(config.precision_output_file, "w")
  precision_readings_file = open(config.precision_readings_output_file, "w")

  for i in range(user_count):
    users_retrieved_docs_count = float(len(test[i]))
    
    if(users_retrieved_docs_count < 1.0):
     continue
    
    included_users += 1
    pred_matches_count = 0.0
    test_users_doc = test[i]
    test_users_set = set(test_users_doc)
    pred_users_doc = pred[i]

    for pd in pred_users_doc:
      if (pd in test_users_set):
        pred_matches_count += 1

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Precision = |{relevant docs} intersection {retrieved docs}|	/	{retrieved docs}
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    user_precision = pred_matches_count / users_retrieved_docs_count

    if (user_precision > max_precision):
      max_precision = user_precision

    print(' u ' + str(i) + '- p ' + str(user_precision))
    precision_file.write(' u {0} - p {1}  ::    {2}  /  {3}\n'.format(i, user_precision, pred_matches_count , users_retrieved_docs_count))
    precision_readings_file.write('{0}\n'.format(user_precision))
    final_precision += user_precision

  print('-' * 50)
  print('final Precision / included users')
  print('{0} / {1}'.format(final_precision, included_users))
  final_precision = final_precision / included_users

  print('Total users - {0}'.format(user_count))
  print('Users included - {0}'.format(included_users))
  print('Max Precision : ' + str(max_precision))
  print('Final Precision : ' + str(final_precision))

  precision_file.write('final Precision / included users\n')
  precision_file.write('{0} / {1}\n'.format(final_precision, included_users))
  precision_file.write('Total users - {0}\n'.format(user_count))
  precision_file.write('Users included - {0}\n'.format(included_users))
  precision_file.write('Max Precision : {0}\n'.format(max_precision))
  precision_file.write('Final Precision : {0}\n'.format(final_precision))
  precision_file.close()
  precision_readings_file.close()



def precision_M(pred, test):
  if (len(pred) != len(test)):
    print("Lengths of predicted & test users dictionary doesnt match by row Count")
    return

  user_count = len(pred)
  final_precision_M = 0.0
  max_precision_M = 0.0
  included_users = 0

  precision_M_file = open(config.precision_M_output_file, "w")
  precision_M_readings_file = open(config.precision_M_readings_output_file, "w")

  for i in range(user_count):
    users_retrieved_docs_count = float(len(test[i]))
    included_users += 1
    pred_matches_count = 0.0
    test_users_doc = test[i]
    test_users_set = set(test_users_doc)
    pred_users_doc = pred[i]

    for pd in pred_users_doc:
      if (pd in test_users_set):
        pred_matches_count += 1

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # precision_M = |{relevant docs} intersection {retrieved docs}|	/	M
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    user_precision_M = pred_matches_count / config.M

    if (user_precision_M > max_precision_M):
      max_precision_M = user_precision_M

    print(' u ' + str(i) + '- p ' + str(user_precision_M))
    precision_M_file.write(' u {0} - p {1}  ::    {2}  /  {3}\n'.format(i, user_precision_M, pred_matches_count , users_retrieved_docs_count))
    precision_M_readings_file.write('{0}\n'.format(user_precision_M))
    final_precision_M += user_precision_M

  print('-' * 50)
  print('final precision_M / included users')
  print('{0} / {1}'.format(final_precision_M, included_users))
  final_precision_M = final_precision_M / included_users

  print('Total users - {0}'.format(user_count))
  print('Users included - {0}'.format(included_users))
  print('Max precision_M : ' + str(max_precision_M))
  print('Final precision_M : ' + str(final_precision_M))

  precision_M_file.write('final precision_M / included users\n')
  precision_M_file.write('{0} / {1}\n'.format(final_precision_M, included_users))
  precision_M_file.write('Total users - {0}\n'.format(user_count))
  precision_M_file.write('Users included - {0}\n'.format(included_users))
  precision_M_file.write('Max precision_M : {0}\n'.format(max_precision_M))
  precision_M_file.write('Final precision_M : {0}\n'.format(final_precision_M))
  precision_M_file.close()
  precision_M_readings_file.close()



def recall(pred, test):
  if (len(pred) != len(test)):
    print("Lengths of predicted & test users dictionary doesnt match by row Count")
    return

  user_count = len(pred)
  final_recall = 0.0
  max_recall = 0.0
  included_users = 0

  recall_file = open(config.recall_output_file, "w")
  recall_readings_file = open(config.recall_readings_output_file, "w")

  for i in range(user_count):
    users_likes_count = float(len(pred[i]))

    if (users_likes_count < 1):
      continue

    included_users += 1
    pred_matches_count = 0.0
    test_users_doc = test[i]
    test_users_set = set(test_users_doc)
    pred_users_doc = pred[i]

    users_relevant_docs_count = float(len(test[i]))

    if(users_relevant_docs_count < 1.0):
      continue
      
    for pd in pred_users_doc:
      if (pd in test_users_set):
        pred_matches_count += 1


    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
    # Recall = |{relevant docs} intersection {retrieved docs}|	/	{relevant docs}
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
   
    user_recall = pred_matches_count / users_relevant_docs_count

    if (user_recall > max_recall):
      max_recall = user_recall

    print(' u ' + str(i) + '- p ' + str(user_recall))
    recall_file.write(' u {0} - p {1}  ::    {2}  /  {3}\n'.format(i, user_recall, pred_matches_count , users_relevant_docs_count))
    recall_readings_file.write('{0}\n'.format(user_recall))
    final_recall += user_recall

  print('-' * 50)
  print('final recall / included users')
  print('{0} / {1}'.format(final_recall, included_users))
  final_recall = final_recall / included_users

  print('Total users - {0}'.format(user_count))
  print('Users included - {0}'.format(included_users))
  print('Max recall : ' + str(max_recall))
  print('Final recall : ' + str(final_recall))

  recall_file.write('final recall / included users\n')
  recall_file.write('{0} / {1}\n'.format(final_recall, included_users))
  recall_file.write('Total users - {0}\n'.format(user_count))
  recall_file.write('Users included - {0}\n'.format(included_users))
  recall_file.write('Max recall : {0}\n'.format(max_recall))
  recall_file.write('Final recall : {0}\n'.format(final_recall))
  recall_file.close()
  recall_readings_file.close()


def process():
  X_pred, X_test = read_predictions_and_test()
  precision(X_pred, X_test)
  precision_M(X_pred, X_test)
  recall(X_pred, X_test)



########################################################################
# All below must be commented, debug only
########################################################################


# process()


########################################################################
