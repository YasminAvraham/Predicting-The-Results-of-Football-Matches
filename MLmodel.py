import sqlite3
import pandas as pd
import numpy as np
import os.path
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from time import time
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings
warnings.simplefilter("ignore")
#check
# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- Gets players data (fifa data) for all matches ---------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_football_matches_data(matches, player_skills):
    print("Collecting players skills data for each match...")
    start = time()
    # get all players skills for each match
    matches_total_data = matches.apply(lambda x: get_players_match_skills(x, player_skills), axis=1)
    end = time()
    print("Players skills for each match collected in {:.1f} minutes".format((end - start) / 60))
    # Return fifa_data
    return matches_total_data

#Aggregates players skills for a given match.
def get_players_match_skills(match, players_skills):
    # Define variables
    match_id = match.match_api_id
    date = match['date']
    players = ['home_player_1', 'home_player_2', 'home_player_3', "home_player_4", "home_player_5",
               "home_player_6", "home_player_7", "home_player_8", "home_player_9", "home_player_10",
               "home_player_11", "away_player_1", "away_player_2", "away_player_3", "away_player_4",
               "away_player_5", "away_player_6", "away_player_7", "away_player_8", "away_player_9",
               "away_player_10", "away_player_11"]
    players_update_skills = pd.DataFrame()
    names = []
    # Loop through all players
    for player in players:
        # Get player ID
        player_id = match[player]
        # Get player skills
        player_skills = players_skills[players_skills.player_api_id == player_id]
        # get the last update skills
        player_skills = player_skills[player_skills.date < date].sort_values(by='date', ascending=False)[:1]
        if np.isnan(player_id) == True:
            overall_rating = pd.Series(0)
        else:
            player_skills.reset_index(inplace=True, drop=True)
            #The total ranking skills of player
            overall_rating = pd.Series(player_skills.loc[0, "overall_rating"])
        # Rename stat
        name = "{}_overall_rating".format(player)
        names.append(name)
        players_update_skills = pd.concat([players_update_skills, overall_rating], axis=1)

    players_update_skills.columns = names
    players_update_skills['match_api_id'] = match_id

    players_update_skills.reset_index(inplace=True, drop=True)

    # Return player stats
    return players_update_skills.ix[0]

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- Get overall players rankings --------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_overall_ranking_skills(matches_players_data, get_overall=False):
    temp_data = matches_players_data
    # Check if only overall player stats are desired
    if get_overall == True:
        # Get overall stats
        data = temp_data.loc[:, (matches_players_data.columns.str.contains('overall_rating'))]
        data.loc[:, 'match_api_id'] = temp_data.loc[:, 'match_api_id']
    else:
        # Get all stats except for stat date
        cols = matches_players_data.loc[:, (matches_players_data.columns.str.contains('date_stat'))]
        temp_data = matches_players_data.drop(cols.columns, axis=1)
        data = temp_data
    # Return data
    return data

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------- Get the last x matches of a given team ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_last_matches(matches, date, team, x=10):
    # Filter team matches from matches
    team_matches = matches[(matches['home_team_api_id'] == team) | (matches['away_team_api_id'] == team)]
    # Filter x last matches from team matches
    last_matches = team_matches[team_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    # Return last matches
    return last_matches

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------- Get the last x matches of two given teams ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_last_matches_against_eachother(matches, date, home_team, away_team, x=10):
    # Find matches of both teams
    home_matches = matches[(matches['home_team_api_id'] == home_team) & (matches['away_team_api_id'] == away_team)]
    away_matches = matches[(matches['home_team_api_id'] == away_team) & (matches['away_team_api_id'] == home_team)]
    total_matches = pd.concat([home_matches, away_matches])
    # Get last x matches
    try:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[0:x, :]
    except:
        last_matches = total_matches[total_matches.date < date].sort_values(by='date', ascending=False).iloc[
                       0:total_matches.shape[0], :]
        # Check for error in data
        if (last_matches.shape[0] > x):
            print("Error in obtaining matches")

    return last_matches

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------- Get the goals of a specfic team from a set of matches ----------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_goals(matches, team):
    # Find home and away goals
    home_goals = int(matches.home_team_goal[matches.home_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.away_team_api_id == team].sum())
    total_goals = home_goals + away_goals
    # Return total goals
    return total_goals

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------- Get the goals conceided of a specific team from a set of matches ---------------------
# ----------------------------------------------------------------------------------------------------------------------


def get_goals_conceided(matches, team):
    # Find home and away goals
    home_goals = int(matches.home_team_goal[matches.away_team_api_id == team].sum())
    away_goals = int(matches.away_team_goal[matches.home_team_api_id == team].sum())
    total_goals = home_goals + away_goals
    # Return total num of goals
    return total_goals

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------- Get the number of wins of a specific team from a set of matches-----------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_wins(matches, team):
    # Find home and away wins
    home_wins = int(matches.home_team_goal[
                        (matches.home_team_api_id == team) & (matches.home_team_goal > matches.away_team_goal)].count())
    away_wins = int(matches.away_team_goal[
                        (matches.away_team_api_id == team) & (matches.away_team_goal > matches.home_team_goal)].count())
    total_wins = home_wins + away_wins
    # Return total wins
    return total_wins

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------Create match specific features for a given match -----------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_match_features(match, matches, x=10):
    # Define variables
    date = match.date
    home_team = match.home_team_api_id
    away_team = match.away_team_api_id
    # Get last x matches of home and away team
    matches_home_team = get_last_matches(matches, date, home_team, x=10)
    matches_away_team = get_last_matches(matches, date, away_team, x=10)
    # Get last x matches of both teams against each other
    last_matches_against = get_last_matches_against_eachother(matches, date, home_team, away_team, x=4)
    # Create goal variables
    home_goals = get_goals(matches_home_team, home_team)
    away_goals = get_goals(matches_away_team, away_team)
    home_goals_conceided = get_goals_conceided(matches_home_team, home_team)
    away_goals_conceided = get_goals_conceided(matches_away_team, away_team)
    # Define result data frame
    result = pd.DataFrame()
    # Define ID features
    result.loc[0, 'match_api_id'] = match.match_api_id
    result.loc[0, 'league_id'] = match.league_id
    # Create match features
    result.loc[0, 'home_team_goals_difference'] = home_goals - home_goals_conceided
    result.loc[0, 'away_team_goals_difference'] = away_goals - away_goals_conceided
    result.loc[0, 'games_won_home_team'] = get_wins(matches_home_team, home_team)
    result.loc[0, 'games_won_away_team'] = get_wins(matches_away_team, away_team)
    result.loc[0, 'games_against_won'] = get_wins(last_matches_against, home_team)
    result.loc[0, 'games_against_lost'] = get_wins(last_matches_against, away_team)
    # Return match features
    return result.loc[0]

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------Derives a label for a given match -----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def get_match_label_result(match):
    # Define variables
    home_goals = match['home_team_goal']
    away_goals = match['away_team_goal']

    label = pd.DataFrame()
    label.loc[0, 'match_api_id'] = match['match_api_id']

    # Identify match label
    if home_goals > away_goals:
        label.loc[0, 'label'] = "Win"
    if home_goals == away_goals:
        label.loc[0, 'label'] = "Draw"
    if home_goals < away_goals:
        label.loc[0, 'label'] = "Defeat"

    # Return label
    return label.loc[0]

#----------------------------------------------------------------------------------------------------------------------
#-------------------------------------------Converts site odds to probabilities. --------------------------------------
#----------------------------------------------------------------------------------------------------------------------

def convert_odds_to_prob(match_odds):
    # Define variables
    match_id = match_odds.loc[:, 'match_api_id']
    site_odd = match_odds.loc[:, 'site_odd']
    win_odd = match_odds.loc[:, 'Win']
    draw_odd = match_odds.loc[:, 'Draw']
    loss_odd = match_odds.loc[:, 'Defeat']
    # Converts odds to prob
    win_prob = 1 / win_odd
    draw_prob = 1 / draw_odd
    loss_prob = 1 / loss_odd
    total_prob = win_prob + draw_prob + loss_prob
    probs = pd.DataFrame()

    # Define output format and scale probs by sum over all probs
    probs.loc[:, 'match_api_id'] = match_id
    probs.loc[:, 'site_odd'] = site_odd
    probs.loc[:, 'Win'] = win_prob / total_prob
    probs.loc[:, 'Draw'] = draw_prob / total_prob
    probs.loc[:, 'Defeat'] = loss_prob / total_prob

    # Return probs and meta data
    return probs
#-----------------------------------------------------------------------------------------------------------------------
#---------------------- Aggregates bet sites odds data for all matches and sites ---------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def get_bets_odds_data(matches, bet_sites_selected, horizontal=True):
    odds_data = pd.DataFrame()
    # Loop through bet sites
    for bet_site_odd in bet_sites_selected:

        # Find columns containing data of site odds
        temp_data = matches.loc[:, (matches.columns.str.contains(bet_site_odd))]
        temp_data.loc[:, 'site_odd'] = str(bet_site_odd)
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']

        # Rename odds columns and convert to numeric
        cols = temp_data.columns.values
        cols[:3] = ['Win', 'Draw', 'Defeat']
        temp_data.columns = cols
        temp_data.loc[:, 'Win'] = pd.to_numeric(temp_data['Win'])
        temp_data.loc[:, 'Draw'] = pd.to_numeric(temp_data['Draw'])
        temp_data.loc[:, 'Defeat'] = pd.to_numeric(temp_data['Defeat'])

        # Check if data should be aggregated horizontally
        if (horizontal == True):
            # Convert data to probs
            temp_data = convert_odds_to_prob(temp_data)
            temp_data.drop('match_api_id', axis=1, inplace=True)
            temp_data.drop('site_odd', axis=1, inplace=True)
            # Rename columns with bookkeeper names
            win_name = bet_site_odd + "_" + "Win"
            draw_name = bet_site_odd + "_" + "Draw"
            defeat_name = bet_site_odd + "_" + "Defeat"
            temp_data.columns.values[:3] = [win_name, draw_name, defeat_name]
            # Aggregate data
            odds_data = pd.concat([odds_data, temp_data], axis=1)
        else:
            # Aggregate vertically
            odds_data = odds_data.append(temp_data, ignore_index=True)
    # If horizontal add match api id to data
    if (horizontal == True):
        temp_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
    # Return site_odds data
    return odds_data

#-----------------------------------------------------------------------------------------------------------------------
#--------------------------------Create and aggregate features and labels for all matches-------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def create_features(matches, matches_players_data, bet_sites_selected_cols, get_overall=False, horizontal=True, x=10, verbose=True):
    # Get players skills for features
    players_skills = get_overall_ranking_skills(matches_players_data, get_overall)
    if verbose == True:
        print("Generating match features...")
    start = time()
    # Get match features for all matches
    match_stats = matches.apply(lambda x: get_match_features(x, matches, x=10), axis=1)
    match_stats.drop(['league_id'], inplace=True, axis=1)

    end = time()
    if verbose == True:
        print("Match features generated in {:.1f} minutes".format((end - start) / 60))

    if verbose == True:
        print("Generating match labels...")
    start = time()

    # Create match labels (axis =1 means apply function to each row.)
    labels = matches.apply(get_match_label_result, axis=1)
    end = time()
    if verbose == True:
        print("Match labels generated in {:.1f} minutes".format((end - start) / 60))

    if verbose == True:
        print("Generating bet sites odds data...")
    start = time()

    # Get bet odds for all matches
    bets_odds_data = get_bets_odds_data(matches, bet_sites_selected_cols, horizontal=True)
    bets_odds_data.loc[:, 'match_api_id'] = matches.loc[:, 'match_api_id']
    end = time()
    if verbose == True:
        print("bet sites odds data generated in {:.1f} minutes".format((end - start) / 60))

    # Merges features and labels into one frame
    features = pd.merge(match_stats, players_skills, on='match_api_id', how='left')
    features = pd.merge(features, bets_odds_data, on='match_api_id', how='left')
    last_features = pd.merge(features, labels, on='match_api_id', how='left')

    # fill_nan_values(feables)- choose not to use!

    # Drop NA values
    last_features.dropna(inplace=True)

    # Return preprocessed data
    return last_features

# fill missing values of bet sites odds with average
def fill_nan_values(features):
    features["B365_Win"] = features["B365_Win"].fillna(value=features["B365_Win"].mean())
    features["B365_Draw"] = features["B365_Draw"].fillna(value=features["B365_Draw"].mean())
    features["B365_Defeat"] = features["B365_Defeat"].fillna(value=features["B365_Defeat"].mean())
    features["BW_Win"] = features["BW_Win"].fillna(value=features["BW_Win"].mean())
    features["BW_Draw"] = features["BW_Draw"].fillna(value=features["BW_Draw"].mean())
    features["BW_Defeat"] = features["BW_Defeat"].fillna(value=features["BW_Defeat"].mean())
    features["IW_Win"] = features["IW_Win"].fillna(value=features["IW_Win"].mean())
    features["IW_Draw"] = features["IW_Draw"].fillna(value=features["IW_Draw"].mean())
    features["IW_Defeat"] = features["IW_Defeat"].fillna(value=features["IW_Defeat"].mean())
    features["LB_Win"] = features["LB_Win"].fillna(value=features["LB_Win"].mean())
    features["LB_Draw"] = features["LB_Draw"].fillna(value=features["LB_Draw"].mean())
    features["LB_Defeat"] = features["LB_Defeat"].fillna(value=features["LB_Defeat"].mean())
    return features


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------- Plot confusion Matrix------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------
#----------------------------------------- Create featuers for train/test datasets--------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

def start_create_features():
    # Connecting to database
    path = "D:\dataset"  # Insert path here
    database = path+'database.sqlite'
    connection = sqlite3.connect(database)
    #Extract required data to tables - SQL
    #players football skills data
    player_skills_data = pd.read_sql("SELECT * FROM Player_Attributes;", connection)
    # matches train data- seasons: 2008/2009-2014/2015:
    matches_data = pd.read_sql("SELECT * FROM Match where season is not '2015/2016' ;", connection)

    data_columns = ["country_id", "league_id", "season", "stage", "date", "match_api_id", "home_team_api_id",
            "away_team_api_id", "home_team_goal", "away_team_goal", "home_player_1", "home_player_2",
            "home_player_3", "home_player_4", "home_player_5", "home_player_6", "home_player_7",
            "home_player_8", "home_player_9", "home_player_10", "home_player_11", "away_player_1",
            "away_player_2", "away_player_3", "away_player_4", "away_player_5", "away_player_6",
            "away_player_7", "away_player_8", "away_player_9", "away_player_10", "away_player_11"]

    matches_data.dropna(subset = data_columns, inplace = True)
    # matches_data = matches_data.tail(3000)
    # Generating features, exploring the data, and preparing data for model training.
    # to each match, complete players skills data that play in the same match
    matches_players_train_data = get_football_matches_data(matches_data, player_skills_data)
    #Creating features and labels of for matches based on players skills, bet odds and data of last past games.
    bet_sites_selected_cols = ['B365', 'BW', 'IW','LB']

    #prepare train dataset features
    train_features = create_features(matches_data, matches_players_train_data, bet_sites_selected_cols, get_overall = True)
    train_features.to_csv('train_data.csv', index=False, header=True)

    #prepare test dataset features- seasons: 2015/2016
    test_matches_data = pd.read_sql("SELECT * FROM Match where season is '2015/2016' ;", connection)
    test_matches_data.dropna(subset = data_columns, inplace = True)
    # test_matches_data = test_matches_data.tail(3000)
    matches_players_test_data = get_football_matches_data(test_matches_data, player_skills_data)
    test_features = create_features(test_matches_data, matches_players_test_data, bet_sites_selected_cols, get_overall = True)
    test_features.to_csv('test_data.csv', index=False, header=True)


#-----------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------- Start----------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------

if(not os.path.exists('train_data.csv') and not os.path.exists('test_data.csv')):
    start_create_features()

train_features = pd.read_csv('train_data.csv')
test_features = pd.read_csv('test_data.csv')
labels_train_result = train_features.loc[:,'label']
train_features = train_features.drop('label', axis = 1)
labels_test_result = test_features.loc[:,'label']
test_features = test_features.drop('label', axis = 1)

#---------------------------------------------------------------------------------------------------------------------
#----------------------------------------------- Train models and show results----------------------------------------
#---------------------------------------------------------------------------------------------------------------------

print ("Models training details:")
k_fold = KFold(n_splits=10, shuffle=True,random_state=0)
LGR_chosen = False
KNN_chosen = False
DT_chosen = False
NB_chosen = False
SVM_chosen = False
RF_chosen = False
best_score = 0

#--Logistic Regression--
print("Logistic Regression model:")
clf = LogisticRegression()
scoring ='accuracy'
score = cross_val_score(clf ,train_features,labels_train_result,cv= k_fold,n_jobs=1, scoring= scoring)
average_score_lgr = round(np.mean(score)*100,2)
print("The average score is:", average_score_lgr)
if (average_score_lgr > best_score):
    best_score= average_score_lgr
    LGR_chosen = True
    KNN_chosen = False
    DT_chosen = False
    NB_chosen = False
    SVM_chosen = False
    RF_chosen = False

#--KNN--
print("Knn model:")
clf = KNeighborsClassifier(n_neighbors= 30)
scoring ='accuracy'
score = cross_val_score(clf ,train_features,labels_train_result,cv= k_fold,n_jobs=1, scoring= scoring)
average_score_knn = round(np.mean(score)*100,2)
print("The average score is:", average_score_knn)
if (average_score_knn > best_score):
    best_score= average_score_knn
    LGR_chosen = False
    KNN_chosen = True
    DT_chosen = False
    NB_chosen = False
    SVM_chosen = False
    RF_chosen = False

#--Decision Tree--
print("Decision Tree model:")
clf = DecisionTreeClassifier()
scoring ='accuracy'
score = cross_val_score(clf ,train_features,labels_train_result,cv= k_fold,n_jobs=1, scoring= scoring)
average_score_dt = round(np.mean(score)*100,2)
print("The average score is:", average_score_dt)
if (average_score_dt > best_score):
    best_score= average_score_dt
    LGR_chosen = False
    KNN_chosen = False
    DT_chosen = True
    NB_chosen = False
    SVM_chosen = False
    RF_chosen = False

#--Naive Bayes--
print("Naive Bayes model:")
clf = GaussianNB()
scoring ='accuracy'
score = cross_val_score(clf ,train_features,labels_train_result,cv= k_fold,n_jobs=1, scoring= scoring)
average_score_nb = round(np.mean(score)*100,2)
print("The average score is:", average_score_nb)
if (average_score_nb > best_score):
    best_score= average_score_nb
    LGR_chosen = False
    KNN_chosen = False
    DT_chosen = False
    NB_chosen = True
    SVM_chosen = False
    RF_chosen = False

#--SVM--
print("SVM model:")
clf = SVC(kernel='rbf',C=0.8,gamma=0.4)
scoring ='accuracy'
score = cross_val_score(clf ,train_features,labels_train_result,cv= k_fold,n_jobs=1, scoring= scoring)
average_score_svm = round(np.mean(score)*100,2)
print("The average score is:", average_score_svm)
if (average_score_svm > best_score):
    best_score= average_score_svm
    LGR_chosen = False
    KNN_chosen = False
    DT_chosen = False
    NB_chosen = False
    SVM_chosen = True
    RF_chosen = False

#--Random Forest--
print("Random Forest model:")
clf = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=10, max_features='auto')
scoring ='accuracy'
score = cross_val_score(clf ,train_features,labels_train_result,cv= k_fold,n_jobs=1, scoring= scoring)
average_score_rf = round(np.mean(score)*100,2)
print("The average score is:", average_score_rf)
if (average_score_rf > best_score):
    best_score= average_score_rf
    LGR_chosen = False
    KNN_chosen = False
    DT_chosen = False
    NB_chosen = False
    SVM_chosen = False
    RF_chosen = True

#---------------------------------------------------------------------------------------------------------------------
#----------------------------------------------- plot bar chart of train results--------------------------------------
#---------------------------------------------------------------------------------------------------------------------

labels_classes = ['Defeat','Draw','Win']
#plot models training results 1:
x = ['Logisic Regression', 'KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Random Forest']
y = [average_score_lgr, average_score_knn, average_score_dt, average_score_nb, average_score_svm, average_score_rf]
fig, ax = plt.subplots()
width = 0.5 # the width of the bars
ind = np.arange(len(y))  # the x locations for the groups
ax.barh(ind, y, width, color="blue")
ax.set_yticks(ind+width/2)
ax.set_yticklabels(x, minor=False)
plt.title('Models training results:')
plt.xlabel('Accuracy')
plt.ylabel('Models')
for i, v in enumerate(y):
    ax.text(v+0.2 , i, str(v), color='blue', fontweight='bold')
plt.show()

#plot models training results 2:
objects = ('Logisic Regression', 'KNN', 'Decision Tree', 'Naive Bayes', 'SVM', 'Random Forest')
y_pos = np.arange(len(objects))
performance = [average_score_lgr, average_score_knn, average_score_dt, average_score_nb, average_score_svm, average_score_rf]
plt.bar(y_pos, performance, align='center', alpha=0.2)
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy')
plt.title('Models training results:')
plt.show()

#---------------------------------------------------------------------------------------------------------------------
#--------------------------------------------- Test and envaluate the best model--------------------------------------
#---------------------------------------------------------------------------------------------------------------------

if (LGR_chosen == True):
    print("The chosen model to test is Logisic Regression:")
    model_chosen = LogisticRegression()
    model_chosen.fit(train_features,labels_train_result)
    predictions = model_chosen.predict(test_features)
    print('Accuracy:', metrics.accuracy_score(labels_test_result, predictions))
    print('Classification Report:')
    print(classification_report(labels_test_result, predictions))
    print('Confusion Matrix:')
    cnf_matrix = confusion_matrix(labels_test_result, predictions)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels_classes,
                          title='Confusion matrix, without normalization')


if (KNN_chosen == True):
    print("The chosen model to test is KNN:")
    model_chosen = KNeighborsClassifier(n_neighbors= 30)
    model_chosen.fit(train_features,labels_train_result)
    predictions = model_chosen.predict(test_features)
    print('Accuracy:', metrics.accuracy_score(labels_test_result, predictions))
    print('Classification Report:')
    print(classification_report(labels_test_result, predictions))
    print('Confusion Matrix:')
    cnf_matrix = confusion_matrix(labels_test_result, predictions)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels_classes,
                          title='Confusion matrix, without normalization')

if (DT_chosen == True):
    print("The chosen model to test is Decision Tree:")
    model_chosen = DecisionTreeClassifier()
    model_chosen.fit(train_features,labels_train_result)
    predictions = model_chosen.predict(test_features)
    print('Accuracy:', metrics.accuracy_score(labels_test_result, predictions))
    print('Classification Report:')
    print(classification_report(labels_test_result, predictions))
    print('Confusion Matrix:')
    cnf_matrix = confusion_matrix(labels_test_result, predictions)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels_classes,
                          title='Confusion matrix, without normalization')

if (NB_chosen == True):
    print("The chosen model to test is Naive Bayes:")
    model_chosen = GaussianNB()
    model_chosen.fit(train_features,labels_train_result)
    predictions = model_chosen.predict(test_features)
    print('Accuracy:', metrics.accuracy_score(labels_test_result, predictions))
    print('Classification Report:')
    print(classification_report(labels_test_result, predictions))
    print('Confusion Matrix:')
    cnf_matrix = confusion_matrix(labels_test_result, predictions)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels_classes,
                          title='Confusion matrix, without normalization')

if (SVM_chosen == True):
    print("The chosen model to test is SVM:")
    model_chosen = SVC(kernel='rbf',C=0.24,gamma=0.15)
    model_chosen.fit(train_features,labels_train_result)
    predictions = model_chosen.predict(test_features)
    print('Accuracy:', metrics.accuracy_score(labels_test_result, predictions))
    print('Classification Report:')
    print(classification_report(labels_test_result, predictions))
    print('Confusion Matrix:')
    cnf_matrix = confusion_matrix(labels_test_result, predictions)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels_classes,
                          title='Confusion matrix, without normalization')

if (RF_chosen == True):
    print("The chosen model to test is Random Forest:")
    model_chosen = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=10, max_features='auto')
    model_chosen.fit(train_features,labels_train_result)
    predictions = model_chosen.predict(test_features)
    print('Accuracy:', metrics.accuracy_score(labels_test_result, predictions))
    print('Classification Report:')
    print(classification_report(labels_test_result, predictions))
    print('Confusion Matrix:')
    cnf_matrix = confusion_matrix(labels_test_result, predictions)
    np.set_printoptions(precision=2)
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels_classes,
                          title='Confusion matrix, without normalization')

