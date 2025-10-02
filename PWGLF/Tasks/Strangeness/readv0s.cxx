// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Strangeness reconstruction QA
// =============================
//
// Dedicated task to understand reconstruction
// Special emphasis on PV reconstruction when strangeness is present
// Tested privately, meant to be used on central MC productions now
//
//    Comments, questions, complaints, suggestions?
//    Please write to:
//    david.dobrigkeit.chinellato@cern.ch
//
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "ReconstructionDataFormats/Track.h"
#include "Common/Core/RecoDecay.h"
#include "Common/Core/trackUtilities.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"
#include "PWGLF/DataModel/LFParticleIdentification.h"
#include "Common/Core/TrackSelection.h"
#include "Common/DataModel/McCollisionExtra.h"
#include "Common/DataModel/TrackSelectionTables.h"
#include "Common/DataModel/EventSelection.h"
#include "Common/DataModel/Centrality.h"
#include "Common/DataModel/PIDResponse.h"
#include "DetectorsBase/Propagator.h"
#include "DetectorsBase/GeometryManager.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DataFormatsParameters/GRPMagField.h"
#include "CCDB/BasicCCDBManager.h"

#include "PWGLF/DataModel/LFStrangenessMLTables.h"
#include "PWGLF/DataModel/LFStrangenessPIDTables.h"
#include "PWGLF/DataModel/LFStrangenessTables.h"

#include <TFile.h>
#include <TH2F.h>
#include <TProfile.h>
#include <TLorentzVector.h>
#include <Math/Vector4D.h>
#include <TPDGCode.h>
#include <TDatabasePDG.h>
#include <cmath>
#include <array>
#include <cstdlib>
#include "Framework/ASoAHelpers.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using std::array;

// using MyTracks = soa::Join<aod::Tracks, aod::TracksExtra, aod::pidTPCPr>;
using TracksCompleteIU = soa::Join<aod::TracksIU, aod::TracksExtra, aod::TracksCovIU, aod::TracksDCA>;
using CollisionsWithEvSels = soa::Join<aod::Collisions, aod::EvSels>;

using DauTracks = soa::Join<aod::DauTrackExtras, aod::DauTrackTPCPIDs>;
using V0Candidates = soa::Join<aod::V0CollRefs, aod::V0Cores, aod::V0Extras>;

struct readv0s {
  // one to hold them all
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::AnalysisObject};

  void init(InitContext const&)
  {
    histos.add("hV0s", "hV0s", kTH1F, {{4, 0, 4}});
    histos.get<TH1>(HIST("hV0s"))->GetXaxis()->SetBinLabel(1, "All");
    histos.get<TH1>(HIST("hV0s"))->GetXaxis()->SetBinLabel(2, "Standard V0s");
    histos.get<TH1>(HIST("hV0s"))->GetXaxis()->SetBinLabel(3, "Global tracks");
    histos.get<TH1>(HIST("hV0s"))->GetXaxis()->SetBinLabel(4, "At least 1 non-ITS track");
  }

  void processOriginal(aod::V0s const& v0tables, TracksCompleteIU const&)
  {
    for (auto& v0 : v0tables) {
      histos.fill(HIST("hV0s"), 0.5);
      if(v0.v0Type() != 1)
        continue;

      histos.fill(HIST("hV0s"), 1.5);
      auto posPartTrack = v0.posTrack_as<TracksCompleteIU>();
      auto negPartTrack = v0.negTrack_as<TracksCompleteIU>();

      if (!posPartTrack.hasITS() || !negPartTrack.hasITS())
        histos.fill(HIST("hV0s"), 3.5);
      else
        histos.fill(HIST("hV0s"), 2.5);
    } // end v0 loop
  }

  void processDerived(V0Candidates const& fullV0s, DauTracks const&)
  {
    for (auto& v0 : fullV0s) {
      histos.fill(HIST("hV0s"), 0.5);
      if(v0.v0Type() != 1)
        continue;

      histos.fill(HIST("hV0s"), 1.5);
      auto posPartTrack = v0.posTrackExtra_as<DauTracks>();
      auto negPartTrack = v0.negTrackExtra_as<DauTracks>();

      if (!posPartTrack.hasITS() || !negPartTrack.hasITS())
        histos.fill(HIST("hV0s"), 3.5);
      else
        histos.fill(HIST("hV0s"), 2.5);
    } // end v0 loop
  }
  PROCESS_SWITCH(readv0s, processOriginal, "Do real data analysis", true);
  PROCESS_SWITCH(readv0s, processDerived, "Do real data analysis", false);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<readv0s>(cfgc)};
}