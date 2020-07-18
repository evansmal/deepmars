import { combineReducers } from "redux";

import { State, INITIAL_STATE } from "../state";
import { Action } from "../actions";

export function modelReducer(state: State = INITIAL_STATE, action: Action): State["model"] {
    switch (action.type) {
        case "SET_SELECTED_MODEL":
            return {
                selected_model: action.model
            }
        default:
            return state.model
    }
}

export const allReducers = combineReducers({
    model: modelReducer
})
