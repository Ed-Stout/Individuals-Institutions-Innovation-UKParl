import pandas as pd
import json

speeches = pd.read_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard_speeches_2015-20.csv", dtype=str) #dtype = str so ids match
speeches["speech_order"] = speeches["speech_order"].astype(int) #speeches currently string

ministers = json.load(open(r"G:\My Drive\Birkbeck\Project\Hansard\Source data\ministers-2010.json", encoding="utf-8"))

#========backfill missing person_id from mnis_id=============
people = json.load(open(r"G:\My Drive\Birkbeck\Project\Hansard\Source data\people.json", encoding="utf-8"))

mnis_to_person = {} #datadotparl_id is the MNIS id

for pers in people["persons"]:
    for ident in pers.get("identifiers", []):
        if ident.get("scheme") == "datadotparl_id":
            mnis_to_person[str(ident["identifier"])] = pers["id"]

person_list = speeches["person_id"].tolist()
mnis_list = speeches["mnis_id"].tolist()

filled = []
rescued = 0

#issue with some person_ids missing - fix below
for position in range(len(speeches)):
    person = person_list[position]
    mnis = mnis_list[position]

    if pd.isna(person) and pd.notna(mnis):
        key = str(mnis).split(".")[0]  #mnis_id is float
        if key in mnis_to_person:
            filled.append(mnis_to_person[key])
            rescued += 1
            continue

    filled.append(person)

speeches["person_id"] = filled

print(len(mnis_to_person), "mnis to person_id mappings")
print(rescued, "speeches given a person_id from mnis_id")
print("speeches still with no person_id:", speeches["person_id"].isna().sum())

roles_by_person = {} #dictionary

for record in ministers["memberships"]:
    source = record["source"]

    if source == "datadotparl/committee": #not interested in committee roles at this stage
        continue

    person = record["person_id"] #join later

    end_date = record.get("end_date") #date they left role
    if not end_date:                    #blanks here mean they are still in the role (unlikely due to age of data)
        end_date = "3000-12-31"

    roles = { "start": record["start_date"], #create data
            "end": end_date,
            "role": record["role"],
            "source": source}

    if person not in roles_by_person: #avoid keyerror
        roles_by_person[person] = []
    roles_by_person[person].append(roles)

print(len(roles_by_person), "people with non-committee roles") # check scale

#=====relate posts to each speech date====
person_list = speeches["person_id"].tolist()
date_list = speeches["date"].tolist()

speaker_names = []
gov_names = []
opp_names = []

for position in range(len(speeches)):
    person = person_list[position]
    speech_date = date_list[position]

    speaker_found = []
    gov_found = []
    opp_found = []

    if pd.notna(person) and person in roles_by_person: #person is MP and not null
        for post in roles_by_person[person]:
            if post["start"] <= speech_date <= post["end"]: #inbetween end and start of speech date
                if post["source"] == "datadotparl/governmentpost":
                    gov_found.append(post["role"])
                elif post["source"] == "datadotparl/oppositionpost":
                    opp_found.append(post["role"])
                elif post["source"] == "datadotparl/parliamentarypost":
                    speaker_found.append(post["role"])

    if len(speaker_found) > 0:
        speaker_names.append("; ".join(speaker_found))
    else:
        speaker_names.append(None)

    if len(gov_found) > 0:
        gov_names.append("; ".join(gov_found))
    else:
        gov_names.append(None)

    if len(opp_found) > 0:
        opp_names.append("; ".join(opp_found))
    else:
        opp_names.append(None)

speeches["speaker_role_name"] = speaker_names
speeches["gov_role_name"] = gov_names
speeches["opp_role_name"] = opp_names

#======create categories for MPs - far too many toles to mp atm
chair_roles = ["Speaker of the House of Commons", #speaker roles are important to be removed to align w/ barron
    "Deputy Speaker and Chairman of Ways and Means",
    "Deputy Speaker (First Deputy Chairman of Ways and Means)",
    "Deputy Speaker (Second Deputy Chairman of Ways and Means)"]

role_tiers = []
roles = []

for position in range(len(speeches)):
    speaker_role = speaker_names[position]
    gov_role = gov_names[position]
    opp_role = opp_names[position]

    is_chair = False
    if speaker_role is not None:
        for chair_role in chair_roles:
            if chair_role in speaker_role: #in not == to avoid error on multiple roles
                is_chair = True

    if is_chair:
        role_tiers.append("chair")
        roles.append(speaker_role)
    elif gov_role is not None:
        role_tiers.append("government")
        roles.append(gov_role)
    elif opp_role is not None:
        role_tiers.append("opposition")
        roles.append(opp_role)
    elif speaker_role is not None:      
        role_tiers.append("backbencher")
        roles.append(speaker_role)
    else:
        role_tiers.append("backbencher") #no role, backbencher assumed
        roles.append(None)

speeches["role"] = roles
speeches["role_tier"] = role_tiers

speeches.to_csv(r"G:\My Drive\Birkbeck\Project\Hansard\hansard-speeches-2015_20-step3.csv", index=False)

#==========checks==============
print(len(speaker_names), len(gov_names), len(opp_names), len(speeches), "should all match")
print("speeches with no person_id:", speeches["person_id"].isna().sum())
print("speech_order still sorted:", speeches["speech_order"].is_monotonic_increasing)
print(speeches["role_tier"].value_counts())