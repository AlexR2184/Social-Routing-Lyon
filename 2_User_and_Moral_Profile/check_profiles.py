import pickle

with open("user_profile_4h.pickle","rb") as f:
    user_param_store_stack, moral_user_stack = pickle.load(f)

profiles = sorted(user_param_store_stack.keys())
for scheme in ["Sacrifice","CollectiveGood"]:
    base = user_param_store_stack[profiles[0]][scheme]
    for p in profiles[1:]:
        print(scheme, "profile", p, "params_identical_to_profile0? ",
              user_param_store_stack[p][scheme] == base)

base_m = moral_user_stack[profiles[0]]
for p in profiles[1:]:
    print("morals profile", p, "identical_to_profile0? ", moral_user_stack[p] == base_m)
